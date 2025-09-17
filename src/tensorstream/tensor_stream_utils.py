import itertools
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .tensor_stream import Event, ModalityType, Stream, TensorStream, create_stream, group_streams


def compute_mrope_pos_tensor(ts: TensorStream, n_pos_dims: int = 3) -> torch.Tensor:
    """
    Create a (batch, T, n_pos_dims) position tensor in one sweep.
    The first dim is the running “time” index, the rest are spatial (or 1-fillers).

    Args:
        ts         : TensorStream
        n_pos_dims : total coordinate dimensions (default 3)

    Returns:
        torch.LongTensor  - shape (batch_size, seq_len, n_pos_dims)
    """

    # Manually iterate through streams and events like map_compact does,
    # but maintain cumulative time offset for each stream
    all_coords = []
    for stream in ts.streams:  # one Stream == one batch sample
        cumulative_offset = 0  # running time index for this stream

        for event in stream:
            # --- build coordinate grid for THIS event using itertools (no tensor ops) ---
            dims = (event.dims() or [1]) + [1] * (n_pos_dims - len(event.dims() or []))

            # Create ranges for each dimension (similar to old _finalize implementation)
            first_dim = range(cumulative_offset, cumulative_offset + dims[0])
            cumulative_offset += dims[0]  # advance time for the next event
            other_dims = [range(d) for d in dims[1:]]

            # Use itertools.product to create all coordinate combinations
            full_coords = list(itertools.product(first_dim, *other_dims))

            # Slice if the event is partial
            s, e = event.idx_range
            coords = full_coords[s:e]

            # Extend the flattened coordinate list
            all_coords.extend(coords)

    # Convert to tensor and reshape to (B, T, n_pos_dims)
    B, T = ts.shape
    return torch.tensor(all_coords, dtype=torch.long, device=ts.device).reshape(B, T, n_pos_dims)


# ──────────────────────────────────────────────────────────────────────────
# Generic event-labelling helper
# ──────────────────────────────────────────────────────────────────────────
def event_mask(
    ts: TensorStream,
    tag_fn: Callable[[Event], int | None],
    default: int = -1,
) -> torch.Tensor:
    """
    Build a (batch, seq_len) LongTensor whose value for every *token*
    is given by `tag_fn(event)`, falling back to `default` when the
    function returns None.

    The work is done in a single pass via `map  →  compact`.
    """

    def to_label(ev: Event) -> Any:
        label = tag_fn(ev)
        if label is None:
            label = default
        return [label] * ev.num_tokens()

    return ts.map_compact(to_label).squeeze(-1)


def event_mask_by_key(
    ts: TensorStream,
    key: str,
    tag_index: dict[str, int],
    default: int = -1,
) -> torch.Tensor:
    """
    Faster call-site syntax when you just want to look up
    `event.tags[key]` and map it through `tag_index`.
    """
    return event_mask(
        ts,
        lambda ev: tag_index.get(ev.tags.get(key)) if key in ev.tags else None,
        default=default,
    )


def modality_mask(ts: TensorStream) -> torch.Tensor:
    return event_mask(ts, lambda ev: ev.type.value)


ROLE_TO_IDX = {
    None: -1,
    "": -1,
    "agent": 0,
    "user": 1,
    "system": 2,
    # … add more if you like
}


def role_mask(ts: TensorStream) -> torch.Tensor:
    return event_mask(ts, lambda ev: ROLE_TO_IDX.get(ev.role, -1))


def tensor_stream_token_view(ts: TensorStream) -> torch.Tensor:
    """
    Return a (B, T) token view by summing across the last dim of every
    event and flattening over the selected token range.
    """

    def to_token_view(ev: Event) -> list[int]:
        # collapse all but the last dim, cast to long
        flat = ev.data.sum(dim=-1).long().reshape(-1)
        if ev.idx_range is not None:
            s, e = ev.idx_range
            return flat[s:e].tolist()
        else:
            return flat.tolist()

    return ts.map_compact(to_token_view)  # shape (B, T)


def reconstruct_tensor_stream_from_compact_dict(
    ts: TensorStream, compact_dict: dict[ModalityType, torch.Tensor]
) -> TensorStream:
    streams = []
    for stream in ts.streams:
        event_list = []
        for event in stream:
            new_event = event.shallow_copy()
            new_event.data = compact_dict[event.type][event.idx_range[0] : event.idx_range[1]]
            compact_dict[event.type] = compact_dict[event.type][event.num_tokens(partial=False) :]
            event_list.append(new_event)
        streams.append(Stream(event_list, priority=stream.priority))
    return TensorStream(streams)


def set_data(
    tensor_stream: TensorStream,
    stream_types: Iterable[ModalityType],
    roles: Iterable[str] = ROLE_TO_IDX.keys(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gathers data from a TensorStream according to the given stream types
    and returns (data, mask) where 'data' has valid entries for
    each requested stream type and 'mask' indicates which elements
    in 'data' are valid.

    NOTE: Currently assumes stream_types are text-based types, but can be extended.

    Args:
        tensor_stream (TensorStream):
            The input TensorStream which contains data for multiple modalities.
        stream_types (Iterable[ModalityType]):
            A list or iterable of modality types (e.g., TextType, VisionType, etc.)
            to retrieve from the TensorStream.
        exclude_non_agent_roles (bool, optional):
            If True, only include tokens with role="agent" or role=None in the loss calculation.
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - data: A tensor of the same shape as the internal metadata shape,
              containing valid entries from the given stream types.
            - mask: A boolean tensor of the same shape, where True indicates
              the corresponding element in 'data' is valid/used.
    """
    # Retrieve indexing and shape metadata
    st_tensor = modality_mask(tensor_stream)  # (B, T) modality-ids
    roles_tensor = role_mask(tensor_stream)  # (B, T) role-ids

    # Create output data placeholders on the same device
    data = torch.zeros_like(st_tensor).to(tensor_stream.device)
    set_data_mask = torch.zeros_like(st_tensor).bool().to(tensor_stream.device).bool()
    per_modality_stream = group_streams(tensor_stream.flat_stream(), group_fn=lambda ev: ev.type, schedule=False)
    per_modality_compact_stream = {k: v.compact() for k, v in per_modality_stream.items()}

    # Fill 'data' and 'set_data_mask' for each requested stream type
    for st in stream_types:
        data_mask = st_tensor == st.value
        partial_mask = (
            per_modality_stream[st]
            .map_compact(
                lambda ev: [int(ev.idx_range[0] <= i < ev.idx_range[1]) for i in range(ev.num_tokens(partial=False))]
            )
            .bool()
        )
        data[data_mask] = per_modality_compact_stream[st].reshape(-1)[partial_mask]

        roles_mask = torch.zeros_like(st_tensor).bool().to(tensor_stream.device).bool()
        for role in roles:
            roles_mask |= roles_tensor == ROLE_TO_IDX[role]
        data_mask = data_mask & roles_mask
        set_data_mask[data_mask] = True

    return data, set_data_mask


def slice(tensor_stream: TensorStream, start: int, end: int) -> TensorStream:
    """
    Return a new TensorStream that contains *only* the tokens in the
    half-open interval ``[start, end)`` (0-based, inclusive-exclusive).
    """
    B, T = tensor_stream.shape
    assert 0 <= start <= end <= T, f"slice [{start}, {end}) is out of bounds for sequence length {T}"

    sliced_streams: list[Stream] = []

    for stream in tensor_stream.streams:
        # current position in tensor stream token dims
        curr_global_index = 0
        new_events: list[Event] = []

        # iterate over each of the events in the stream only selecting
        # the events that fall within the range
        for ev in stream:
            ev_len = ev.num_tokens()

            # ev_start, ev_end are the start and end indicies of the
            # event within the tensor stream token dim
            global_ev_start, global_ev_end = curr_global_index, curr_global_index + ev_len

            if global_ev_end <= start:
                # The event occurs before the start skip it and move the cursor
                # forward
                curr_global_index = global_ev_end
                continue
            if global_ev_start >= end:
                # event occurs after the end we can exit
                break

            # only consider the part of the event that falls within the range
            keep_from = max(0, start - global_ev_start)
            keep_to = min(ev_len, end - global_ev_start)
            part = ev.shallow_copy()

            if keep_from == 0 and keep_to == ev_len:
                # Event lies wholly inside the slice
                new_events.append(part)
            else:
                # Partial overlap → trim.
                assert ev.is_measured

                # update the local event ranges for the slices
                sliced_event_start = part.idx_range[0] + keep_from
                sliced_event_end = part.idx_range[0] + keep_to
                part.slice_tokens(sliced_event_start, sliced_event_end)
                new_events.append(part)

            curr_global_index = global_ev_end

        sliced_streams.append(create_stream(new_events, stream.priority, schedule=False))

    return TensorStream(sliced_streams)
