import functools

import pytest
import torch

from src.preprocess.target_processor import TargetProcessor

assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
def test_start_end_samples_to_targets():
    target_processor = TargetProcessor(
        sample_rate=8000,
        win_length=8000,
        hop_length=4000,
    )
    target_processor.create_empty_targets(size=8)
    output = target_processor.start_end_samples_to_targets(16902, 32431)
    expected_output = torch.Tensor(
        [0.0000, 0.0000, 0.0000, 0.3872, 0.8872, 1.0000, 1.0000, 0.5539]
    )
    assert_equal(output, expected_output)
