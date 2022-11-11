import torchaudio.functional as F
import torchaudio.transforms as T
import torchaudio
import constants
import logging
import IPython.display as ipd


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Preprocessor:
    def __init__(self):
        self.sample_rate = None

    def preprocess(self, waveform):
        pass

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        self.sample_rate = sample_rate
        return waveform

    def cut_audio(self, waveform, start_second, end_second):
        start_sample = (self.sample_rate * start_second)
        end_sample = (self.sample_rate * end_second)
        return waveform[0, start_sample:end_sample].unsqueeze(dim=0)

    def play_audio(self, waveform):
        ipd.Audio(waveform.numpy(), rate=self.sample_rate)

    def resample(self, waveform, new_sample_rate):
        waveform = F.resample(
            waveform, orig_freq=self.sample_rate, new_freq=new_sample_rate
        )
        logger.debug(f"Sample rate from {self.sample_rate} to {new_sample_rate}.")
        self.sample_rate = new_sample_rate
        return waveform

    def mfcc(self, waveform, n_mfcc=40):
        mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': constants.WIN_LENGTH,
                'hop_length': constants.HOP_LENGTH,
                'mel_scale': 'htk',
            }
        )
        return mfcc_transform(waveform)
