import os

import torchaudio.functional as F
import torchaudio.transforms as T
import torchaudio
import constants
import logging
import IPython.display as ipd


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

N_MFCC = 40


class FeatureExtractor:
    def __init__(self):
        self.sample_rate = None

    def process_file(self, rel_path):
        """
        Read one audio file from path and apply all preprocessing.

        Parameters
        ----------
        rel_path: Relative path to the audio. (e.g. data/audio/speaker.wav)

        Returns
        -------
        Audio transformed into MFCC coefficients.
        """
        abs_path = os.path.join(constants.BASE_PATH, rel_path)
        waveform = self.load_audio(abs_path)
        waveform = self.resample(waveform, new_sample_rate=constants.SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = FeatureExtractor.stereo_to_mono(waveform)
        features = self.mfcc(waveform)[0]
        return features

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        self.sample_rate = sample_rate
        return waveform

    @staticmethod
    def stereo_to_mono(waveform):
        """Turn stereo audio into mono."""
        return waveform.mean(dim=0).unsqueeze(dim=0)

    def cut_audio(self, waveform, start_second, end_second):
        start_sample = self.sample_rate * start_second
        end_sample = self.sample_rate * end_second
        return waveform[0, start_sample:end_sample].unsqueeze(dim=0)

    def play_audio(self, waveform):
        return ipd.Audio(waveform.numpy(), rate=self.sample_rate)

    def resample(self, waveform, new_sample_rate):
        waveform = F.resample(
            waveform, orig_freq=self.sample_rate, new_freq=new_sample_rate
        )
        logger.debug(f"Sample rate from {self.sample_rate} to {new_sample_rate}.")
        self.sample_rate = new_sample_rate
        return waveform

    def mfcc(self, waveform, n_mfcc=N_MFCC):
        mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": constants.WIN_LENGTH,
                "hop_length": constants.HOP_LENGTH,
                "mel_scale": "htk",
            },
        )
        return mfcc_transform(waveform)
