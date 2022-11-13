import logging
import os

import pandas as pd
import torch

import constants

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class TargetProcessor:
    def __init__(
        self,
        sample_rate=constants.SAMPLE_RATE,
        win_length=constants.WIN_LENGTH,
        hop_length=constants.HOP_LENGTH,
    ):
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.hops_per_window = win_length / hop_length
        self.targets = None
        self.frames = None

    def process(self, rel_path, size):
        """
        Transform annotations in form of start and end time (in seconds) into the
        amount of contamination per frame.

        Parameters
        ----------
        rel_path: Relative path to annotation file.
        size: Size of target array. Depends on X.

        Returns
        -------
        Array of targets with values [0.0, 1.0].
        """
        self.reset()
        self.create_empty_targets(size)
        start_samples, end_samples = self.read_annotations(rel_path)
        for start, end in zip(start_samples, end_samples):
            self.start_end_samples_to_targets(start, end)
        return self.targets

    def reset(self):
        self.targets = None
        self.frames = None

    def create_empty_targets(self, size):
        # target_size = math.ceil(
        #     (waveform.shape[1] - (self.win_length - self.hop_length)) / self.hop_length
        # )
        self.targets = torch.zeros(size)
        self.frames = torch.arange(size)

    def start_end_samples_to_targets(self, start_sample, end_sample):
        """
        Transforms filler starting and ending time into the amount of contamination
        per frame. The contamination will range from 0.0 to 1.0.

        Parameters
        ----------
        start_sample: Sample at which the filler sound starts
        end_sample: Sample at which the filler sound ends

        Returns
        -------
        Array with targets that contains the current filler.
        """
        start_frame = start_sample / self.hop_length
        end_frame = end_sample / self.hop_length
        filler_length_frames = end_frame - start_frame

        mask = (self.frames + self.hops_per_window > start_frame) & (
            self.frames <= end_frame
        )

        filler_start_to_frame_start = self.frames[mask] - start_frame
        filler_end_to_frame_end = end_frame - self.frames[mask] - self.hops_per_window
        filler_start_to_frame_start[filler_start_to_frame_start < 0] = 0
        filler_end_to_frame_end[filler_end_to_frame_end < 0] = 0

        contamination = filler_length_frames - (
            filler_start_to_frame_start + filler_end_to_frame_end
        )
        contamination /= self.hops_per_window
        self.targets[mask] += contamination
        return self.targets

    def read_annotations(self, rel_path):
        """Extracts start & end time from an annotation file
        and transforms them into start and end samples."""
        abs_path = os.path.join(constants.BASE_PATH, rel_path)
        df = pd.read_csv(abs_path)
        start_samples = df["start_time"].apply(self.seconds_to_samples).to_numpy()
        end_samples = df["end_time"].apply(self.seconds_to_samples).to_numpy()
        return start_samples, end_samples

    def seconds_to_samples(self, x: float) -> int:
        """Convert seconds into samples."""
        return int(x * self.sample_rate)
