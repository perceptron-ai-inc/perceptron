from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import (
    Any,
    NewType,
)

import torch
from torch.profiler import record_function


class ModalityType(Enum):
    """
    Base class for modality-type enumerations.
    Each derived class (VisionType, TextType) holds
    an integer value that identifies a specific modality.

    Example usage:
        If you have an object `my_event` of class `Event`,
        you might write:
            if my_event.type == VisionType.image:
                # process an image frame

    The methods below implement ordering and hashing
    based on the integer `.value` of each enum member.
    """

    @property
    def modality(self):
        return self.__class__

    def __lt__(self, other):
        if isinstance(other, ModalityType):
            return self.value < other.value
        raise NotImplementedError()

    def __eq__(self, other):
        if isinstance(other, ModalityType):
            return self.value == other.value
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.value)


# NOTE: modality types need to be unique
class VisionType(ModalityType):
    """
    Enum for vision modalities such as key video frames.
    Typically used in video processing or image sequences.

    Members:
        image: A single image frame.
    """

    image = 0


class TextType(ModalityType):
    """
    Enum for text tokens and padding.

    Members:
        text: Actual textual tokens.
        padding: Padding tokens used in sequence batching.
    """

    text = 1
    padding = 2


# maps idx -> type
ALL_TYPES = [
    tp
    for types in [
        list(VisionType),
        list(TextType),
    ]
    for tp in types
]


# @dataclass
@dataclass(slots=True)
class Event:
    """
    Represents a single data occurrence (with a specific type, time interval, and data payload).

    Attributes:
        data (Any): The actual data payload (e.g. a torch.Tensor, a string, etc.).
        type (ModalityType): The modality type of the data (e.g., VisionType.image).
        time (Tuple[float, float]): (start_time, end_time) indicating when this Event occurs.
        role (Optional[str]): The role associated with this event (e.g., "user", "agent", "system").
            If None, the event is always included in loss calculation.

    Example usage:
        evt = Event(data=torch.zeros((1, 224, 224, 3)),  # e.g. a single image frame
                    type=VisionType.image,
                    time=(0.0, 0.04),
                    role="user")
    """

    # Descriptors
    data: Any
    time: tuple[float, float]
    type: ModalityType
    role: str | None = None

    # Structure
    dims_virtual: list[int] | None = None  # virtual/processed dimensions (e.g., pixel-shuffled)
    dims_real: list[int] | None = None  # real/actual tensor dimensions
    idx_range: tuple[int, int] | None = None

    # Misc Tags (data source, shard idx, etc.)
    tags: dict = field(default_factory=dict)

    def dims(self, virtual: bool = True) -> list[int] | None:
        """
        Get the dimensions of this event.

        Args:
            virtual: If True (default), return virtual/processed dimensions (e.g., pixel-shuffled).
                    If False, return real/actual tensor dimensions.

        Returns:
            Dimensions list or None if not measured.
        """
        if virtual:
            return self.dims_virtual
        else:
            return self.dims_real

    @property
    def is_measured(self):
        return self.dims_virtual is not None

    def slice_tokens(self, start: int | None = None, end: int | None = None):
        """
        Converts into a partial event where the only valid data is between start and end indices of the flattened data
        """
        assert self.is_measured
        assert start is not None and end is not None
        assert self.idx_range[0] <= start <= end <= self.idx_range[1]
        self.idx_range = (start or 0, end or math.prod(self.dims()))

    def num_tokens(self, partial=True, virtual=True) -> int:
        if not virtual:
            assert partial is False and isinstance(self.data, torch.Tensor)
            return math.prod(self.dims(virtual=False))
        return self.idx_range[1] - self.idx_range[0] if partial else math.prod(self.dims())

    def shallow_copy(self) -> Event:
        return replace(self)

    def __hash__(self) -> int:
        """Hash Event based on structure, excluding data."""

        def make_hashable(obj):
            """Convert any object to hashable form."""
            if obj is None:
                return None
            elif isinstance(obj, str | int | float | bool | tuple):
                return obj
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj) if obj else None
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items())) if obj else None
            elif hasattr(obj, "value"):  # Enum types
                return obj.value
            else:
                return str(obj)  # Fallback for other types

        hash_values = []
        for fld in fields(self):
            if fld.name == "data":
                continue  # Skip tensor data

            value = getattr(self, fld.name)
            hash_values.append(make_hashable(value))

        return hash(tuple(hash_values))

    def __eq__(self, other) -> bool:
        """
        Compares two Event objects for strict equality,
        allowing for float tolerances in torch.Tensors (via torch.allclose).
        """
        if not isinstance(other, Event):
            return False

        for fld in fields(self):
            self_value = getattr(self, fld.name)
            other_value = getattr(other, fld.name)

            if fld.name == "data":
                # Special handling for tensor data with float tolerance
                if isinstance(self_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
                    if not torch.allclose(self_value, other_value):
                        return False
                else:
                    if self_value != other_value:
                        return False
            elif fld.name == "role":
                # Special handling for role: both must be None or both must be set and equal
                if (self_value is None) != (other_value is None):
                    return False
                if self_value is not None and self_value != other_value:
                    return False
            else:
                # Standard equality for all other fields
                if self_value != other_value:
                    return False

        return True


@dataclass
class Stream:
    """
    Represents an ordered sequence of Event objects, each with
    a specific ModalityType and a time range.

    Attributes:
        events (List[Event]): The list of Event objects in the stream.
        priority (List[ModalityType]): A list of modality types that define
            how we might want to reorder or prioritize events if scheduling is needed.

    Example usage:
        # Create two events of different types
        evt1 = Event(torch.zeros((1, 224, 224, 3)), VisionType.image, (0.0, 0.04))
        evt2 = Event(torch.randint(0, 1000, (16, 1)), TextType.text, (0.0, 0.32))

        # Make a stream with a given priority
        s = Stream(events=[evt1, evt2],
                   priority=[VisionType.image, TextType.text])

        print(s)
    """

    events: list[Event]
    priority: list[ModalityType]  # priority of stream ordering

    def __len__(self):
        """Returns the number of Event objects in this Stream."""
        return len(self.events)

    def __getitem__(self, key: int) -> Stream | Event:
        return self.events[key]

    def __iter__(self):
        """
        Yields each Event in the Stream, enabling iteration like:
            for event in my_stream:
                ...
        """
        yield from self.events

    # --- after ------------------------------------------------------------
    @record_function("Stream.map")
    def map(
        self,
        func: Callable[[Event], dict[str, Any]],
        *,
        copy_unchanged: bool = False,  # opt-in if you really need isolation
    ) -> Stream:
        """
        Apply *func* to every event and return a new Stream.

        *func* must return a **dict of fields that actually change**.
        We create **one shallow copy** only when something changes;
        unchanged events are reused directly, which is inexpensive and
        keeps autograd graphs intact.
        """
        mapped: list[Event] = []
        for ev in self.events:
            delta = func(ev)
            if not delta:  # fast-path: nothing changes
                mapped.append(ev if not copy_unchanged else ev.shallow_copy())
                continue

            new_ev = ev.shallow_copy()  # ⚡ no tensor clone
            for k, v in delta.items():
                setattr(new_ev, k, v)
            mapped.append(new_ev)

        return create_stream(mapped, priority=self.priority, schedule=False)

    @record_function("Stream.compact")
    def compact(self) -> torch.Tensor:
        assert all([(isinstance(ev.data, torch.Tensor) and ev.is_measured) for ev in self.events]), (
            "Stream.compact only works for streams with events that have measured tensor data"
        )
        return torch.cat([ev.data for ev in self.events]).contiguous()

    @record_function("Stream.map_compact")
    def map_compact(self, event_tf: Callable[[Event], list[Any]]) -> torch.Tensor:
        mapped_list = []
        for event in self:
            mapped_list.extend(event_tf(event))
        tensor = torch.tensor(
            mapped_list,
            dtype=torch.long,
            device=next(
                (ev.data.device for ev in self.events if isinstance(ev.data, torch.Tensor)),
                "cpu",
            ),
        ).contiguous()
        return tensor

    def flatten(self) -> Stream:
        return self.map(lambda ev: {"data": ev.data.reshape(-1, ev.data.shape[-1])})

    def shallow_copy(self) -> Stream:
        events_copy = [ev.shallow_copy() for ev in self.events]
        return create_stream(events=events_copy, priority=self.priority, schedule=False)

    def __hash__(self) -> int:
        """Hash Stream based on structure."""
        return hash(
            (
                tuple(p.value for p in self.priority),  # Convert enums to values
                tuple(hash(event) for event in self.events),  # Use Event.__hash__
            )
        )

    def __eq__(self, other) -> bool:
        """Compare Streams structurally."""
        if not isinstance(other, Stream):
            return False

        return (
            self.priority == other.priority
            and len(self.events) == len(other.events)
            and all(e1 == e2 for e1, e2 in zip(self.events, other.events, strict=False))
        )


# TODO: implement all types of cool indexing which can happen since TensorStream assuems Event.data = Tensor
@dataclass
class TensorStream:
    streams: list[Stream]
    _device: torch.device | None = None

    def __post_init__(self):
        for stream in self.streams:
            for event in stream.events:
                assert isinstance(event.data, torch.Tensor)
                if self._device is None:
                    self._device = torch.device(event.data.device)

    # TODO: implement non-strict compaction modes
    @record_function("TensorStream.compact")
    def compact(self, mode="strict") -> torch.Tensor:
        compact_tensor_stream = torch.stack([stream.compact() for stream in self.streams]).contiguous()
        return compact_tensor_stream

    @record_function("TensorStream.map")
    def map(self, event_tf: Callable[[Event], dict[str, Any]]) -> TensorStream:
        mapped_streams = [stream.map(event_tf) for stream in self.streams]
        return TensorStream(mapped_streams)

    @record_function("TensorStream.map_compact")
    def map_compact(self, event_tf: Callable[[Event], list[Any]]) -> torch.Tensor:
        mapped_list = []
        for stream in self.streams:
            for event in stream:
                mapped_list.extend(event_tf(event))
        B, T = self.shape
        tensor = torch.tensor(mapped_list, dtype=torch.long, device=self.device).reshape(B, T)
        return tensor

    def flat_stream(self) -> Stream:
        if not self.streams:
            return create_stream([], priority=[], schedule=False)
        return create_stream(
            [event for stream in self.streams for event in stream], priority=self.streams[0].priority, schedule=False
        )

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        seq_lens = [sum([ev.num_tokens() for ev in stream]) for stream in self.streams]
        assert all([sl == seq_lens[0] for sl in seq_lens]), (
            f"each stream must have same token count to have a shape: {seq_lens}"
        )
        return (len(seq_lens), seq_lens[0])

    @record_function("TensorStream.to")
    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        non_blocking: bool = True,
    ) -> TensorStream:
        """
        Move **all** `Event.data` tensors to *device*.

        We send each tensor individually instead of the
        flatten → unflatten round-trip:

        * one async H2D copy per tensor (still overlapped when
          `pin_memory=True` is set on the DataLoader),
        * no extra host-side concat, no extra device allocation,
        * `requires_grad` flags are preserved.

        NOTE: textual modalities are always cast to `torch.long`;
        everything else keeps its original
        dtype unless an explicit *dtype* argument is supplied.
        """
        target_device = torch.device(device)

        for stream in self.streams:
            for ev in stream:
                # ------------------------------------------------------------------
                # Decide the dtype for *this* event.
                # ------------------------------------------------------------------
                if ev.type in list(TextType):
                    tgt_dtype = torch.long
                else:
                    tgt_dtype = dtype or ev.data.dtype

                # ------------------------------------------------------------------
                # Perform the device / dtype move.
                # ------------------------------------------------------------------
                # We clone no tensor here; torch will reuse storage
                # if `dtype` and `device` are unchanged.
                moved = ev.data.to(
                    device=target_device,
                    dtype=tgt_dtype,
                    non_blocking=non_blocking,
                )

                # Preserve autograd leaf & grad-enabled state.
                moved.requires_grad_(ev.data.requires_grad)

                ev.data = moved

        # Remember where the whole TensorStream lives now.
        self._device = target_device
        return self

    @record_function("TensorStream.pin_memory")
    def pin_memory(self, non_blocking: bool = True) -> TensorStream:
        """
        Page-lock (aka *pin*) all **CPU** tensors contained in this
        `TensorStream`.  Pinned tensors make subsequent asynchronous
        H2D copies (e.g. inside `TensorStream.to("cuda")`) faster and,
        when used together with a `DataLoader(pin_memory=True)`,
        enable overlap of host-to-device transfers with GPU execution.

        The call is a no-op for tensors that are already on a CUDA /
        MPS / other non-CPU device.

        Parameters
        ----------
        non_blocking : bool, default = True
            Forwarded to `Tensor.pin_memory()`; should almost always
            stay *True* so later `to(device, non_blocking=True)` calls
            can overlap.

        Returns
        -------
        self : TensorStream
            The same object (mutated in-place) to allow call chaining.
        """
        for stream in self.streams:
            for ev in stream:
                if ev.data.device.type == "cpu":
                    # `pin_memory()` clones only when needed
                    pinned = ev.data.pin_memory()  # noqa: F841
                    # NB: pin_memory() preserves dtype/shape/grad/etc.
                    if not non_blocking:
                        # ensure the pinning work is done now
                        torch.cuda.current_stream().synchronize()  # safe on CPU too
                    ev.data = pinned
        # `_device` **stays** the same (still CPU) – no change needed
        return self

    def __hash__(self) -> int:
        """Hash TensorStream based on structure."""
        return hash(
            (
                tuple(hash(stream) for stream in self.streams),  # Use Stream.__hash__
                str(self._device) if self._device else None,
                self.shape,
            )
        )

    def __eq__(self, other) -> bool:
        """Compare TensorStreams structurally."""
        if not isinstance(other, TensorStream):
            return False

        return (
            self._device == other._device
            and self.shape == other.shape
            and len(self.streams) == len(other.streams)
            and all(s1 == s2 for s1, s2 in zip(self.streams, other.streams, strict=False))
        )


def collate_tensor_stream(
    tensor_streams: list[TensorStream],
) -> TensorStream:
    return TensorStream([stream for ts in tensor_streams for stream in ts.streams])


def _schedule_stream(stream: Stream) -> Stream:
    """
    Internal function that reorders (schedules) the events in a Stream
    based on the stream's priority.

    By default, this calls schedule_events(...) and reorders the events accordingly.
    The new ordering is assigned in-place to stream.events.

    Example usage (indirect):
        new_stream = _schedule_stream(old_stream)
    """
    scheduled_inds = schedule_events(stream, priority=stream.priority)
    stream.events = [stream.events[i] for i in scheduled_inds]
    return stream


def create_stream(events: list[Event], priority: list[ModalityType], schedule: bool = True) -> Stream:
    """
    Creates a new Stream with the given events and priority.
    If 'schedule' is True, the events are reordered by calling _schedule_stream.

    Example usage:
        evt1 = Event(torch.zeros(10), TextType.text, (0.0, 1.0))
        evt2 = Event(torch.ones(10), TextType.text, (1.0, 2.0))
        my_stream = create_stream(events=[evt1, evt2],
                                  priority=[TextType.text],
                                  schedule=False)
        print(my_stream)
    """
    stream = Stream(events, priority)
    if schedule:
        stream = _schedule_stream(stream)
    return stream


def merge_streams(streams: Iterable[Stream]) -> Stream:
    """
    Merges multiple Stream objects into one.
    The priority of the merged stream is chosen from the longest priority list among the inputs.
    Stream priorities must be consistent with the chosen priority.

    All events are concatenated, and a new Stream is created (and scheduled).

    Example usage:
        merged = merge_streams([stream1, stream2])
    """
    chosen_priority = max([stream.priority for stream in streams], key=len)
    assert all(
        [str(stream.priority) in str([p for p in chosen_priority if p in stream.priority]) for stream in streams]
    ), "One or more streams has a priority order that doesn't match the merged stream"
    merged_event_list = [ev for stream in streams for ev in stream.events]
    merged_stream = create_stream(merged_event_list, chosen_priority)  # non-root stream creation
    return merged_stream


EventDescriptor = NewType("EventDescriptor", Any)


# NOTE: actually not used now but thought it *might* be useful
def get_stream_descriptor(
    stream: Stream, measure_fn: Callable[[Event], EventDescriptor] = lambda ev: ev.type
) -> set[Any]:
    """
    Create a set of descriptors for each Event in a Stream based on measure_fn.

    measure_fn maps an Event to a descriptive key.
    For example, if events have different data shapes, one might use:
        measure_fn = lambda ev: ev.data.shape
    i.e.
        stream of VisionTypes with tensors of shapes [(1, 3, 3), (1, 3, 3), (1, 4, 4)]
        get_stream_descriptor(stream, measure_fn=lambda t: t.shape) = {(1, 3, 3), (1, 4, 4)}
        now we can pass this into group_streams which will split out vision sub-streams which can be bundled
    Returns:
        A set of descriptors representing the Events in the stream.

    Example usage:
        descriptor = get_stream_descriptor(my_stream, lambda ev: ev.type)
    """
    stream_descriptor = set()
    for ev in stream.events:
        ev_measurement = measure_fn(ev)
        stream_descriptor.add(ev_measurement)
    return stream_descriptor


def group_streams(
    stream: Stream, group_fn: Callable[[Event], EventDescriptor], schedule=True
) -> dict[EventDescriptor, Stream]:
    """
    Splits a single Stream into multiple sub-Streams, grouped by the output of group_fn(event).

    For example, group_fn could be:
        - lambda ev: ev.type
        - lambda ev: ev.type.modality
        - lambda ev: (ev.type.modality, ev.data.shape)

    Returns:
        A dictionary mapping each group key to a Stream of events belonging to that group.
        If 'schedule' is True, each sub-Stream is scheduled via create_stream(..., schedule=True).

    Example usage:
        substreams = group_streams(my_stream, lambda ev: ev.type)
    """
    split_streams: defaultdict[EventDescriptor, list[Event]] = defaultdict(list)
    for ev in stream:
        group = group_fn(ev)
        split_streams[group].append(ev)
    for g, events in split_streams.items():
        split_streams[g] = create_stream(events, stream.priority, schedule=schedule)
    return dict(split_streams)


# Define Category for clarity
Category = NewType("Category", Any)


def schedule_events(stream: Stream, priority: list[Category]) -> list[int]:
    """
    Schedule events based on their start time and priority using a topological sort algorithm.

    The priority list defines the ordering of categories.

    This function:
      1. Pairs each event with its original index.
      2. Sorts events by start time.
      3. Builds a dependency graph based on overlapping events.
      4. Uses a heap to perform a deterministic topological sort with tie-breakers.

    Raises:
        ValueError: If a cycle is detected in the events (i.e., no valid ordering exists).

    Returns:
        List[int]: A list of original indices representing the scheduled order of events.
    """
    priority_index: dict[Category, int] = {category: idx for idx, category in enumerate(priority)}

    # Pair each event metadata with its original index
    events = []
    for i, event in enumerate(stream.events):
        events.append(
            (
                i,
                event.time[0],
                event.time[1],
                event.type,
            )
        )

    sorted_events = sorted(events, key=lambda e: e[1])  # sort by start time
    num_events = len(sorted_events)

    # Build dependency graph
    graph = defaultdict(set)
    indegree = {i: 0 for i in range(num_events)}

    for i in range(num_events):
        idx_i, start_i, end_i, category_i = sorted_events[i]
        prio_i = priority_index[category_i]
        for j in range(i + 1, num_events):
            idx_j, start_j, end_j, category_j = sorted_events[j]
            if start_j >= end_i:
                break
            if end_i > start_j and end_j > start_i:
                prio_j = priority_index[category_j]
                if prio_i < prio_j:
                    graph[i].add(j)
                    indegree[j] += 1
                elif prio_i > prio_j:
                    graph[j].add(i)
                    indegree[i] += 1

    # Use heap for deterministic tie-breakers: (start_time, priority, original_index)
    heap = [
        (
            sorted_events[i][1],
            priority_index[sorted_events[i][3]],
            sorted_events[i][0],
            i,
        )
        for i in range(num_events)
        if indegree[i] == 0
    ]
    heapq.heapify(heap)
    resolved_order = []

    while heap:
        _, _, _, u = heapq.heappop(heap)
        resolved_order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(
                    heap,
                    (
                        sorted_events[v][1],
                        priority_index[sorted_events[v][3]],
                        sorted_events[v][0],
                        v,
                    ),
                )

    if len(resolved_order) != num_events:
        raise ValueError("Cycle detected in events, cannot resolve order")

    return [sorted_events[i][0] for i in resolved_order]
