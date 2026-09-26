import torch

from perceptron.tensorstream.tensorstream import Event, Stream, TensorStream, TextType
from perceptron.tensorstream.ops import compute_mrope_pos_tensor


def test_compute_mrope_pos_tensor_empty_dims_no_shape_mismatch():
    # event.dims() == [] previously produced n_pos_dims + 1 entries
    # due to inconsistent fallback handling across repeated calls.
    event = Event(
        data=torch.zeros(1, dtype=torch.long),
        time=(0.0, 1.0),
        type=TextType.text,
        dims_virtual=[],
        idx_range=(0, 1),
    )
    stream = Stream(events=[event], priority=[TextType.text])
    ts = TensorStream(streams=[stream])

    result = compute_mrope_pos_tensor(ts, n_pos_dims=3)

    assert result.shape == (1, 1, 3)
