import librosa
import librosa.filters
import soundfile as sf
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from process_util.ParamsUtil import ParamsUtil

_mel_basis = None
class AudioUtils():

    def __init__(self):
        self.hp = ParamsUtil()

    def load_wav(self,path, sr):
        return librosa.core.load(path, sr=sr)[0]

    def save_wav(self,wav, path, sr):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, sr, wav.astype(np.int16))

    def save_wavenet_wav(self,wav, path, sr):
        sf.write(path,wav,16000,)

    def preemphasis(self,wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    def inv_preemphasis(self,wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav

    def get_hop_size(self):
        hop_size = self.hp.hop_size
        if hop_size is None:
            assert self.hp.frame_shift_ms is not None
            hop_size = int(self.hp.frame_shift_ms / 1000 * self.hp.sample_rate)
        return hop_size

    def get_hop_size(self):
        hop_size = self.hp.hop_size
        if hop_size is None:
            assert self.hp.frame_shift_ms is not None
            hop_size = int(self.hp.frame_shift_ms / 1000 * hp.sample_rate)
        return hop_size

    def linearspectrogram(self,wav):
        D = self._stft(self.preemphasis(wav, self.hp.preemphasis, self.hp.preemphasize))
        S = self._amp_to_db(np.abs(D)) - self.hp.ref_level_db

        if self.hp.signal_normalization:
            return self._normalize(S)
        return S

    def melspectrogram(self,wav):
        D = self._stft(self.preemphasis(wav, self.hp.preemphasis, self.hp.preemphasize))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hp.ref_level_db

        if self.hp.signal_normalization:
            return self._normalize(S)
        return S


    def _stft(self,y):
        return librosa.stft(y=y, n_fft=self.hp.n_fft, hop_length=self.get_hop_size(), win_length=self.hp.win_size)

    def librosa_pad_lr(self,x, fsize, fshift):
        return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

    # Conversions


    def _linear_to_mel(self,spectogram):
        global _mel_basis
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectogram)

    def _build_mel_basis(self):
        assert self.hp.fmax <= self.hp.sample_rate // 2
        return librosa.filters.mel(self.hp.sample_rate, self.hp.n_fft, n_mels=self.hp.num_mels,
                                   fmin=self.hp.fmin, fmax=self.hp.fmax)

    def _amp_to_db(self,x):
        min_level = np.exp(self.hp.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self,x):
        return np.power(10.0, (x) * 0.05)

    def _normalize(self,S):
        if self.hp.allow_clipping_in_normalization:
            if self.hp.symmetric_mels:
                return np.clip(
                    (2 * self.hp.max_abs_value) * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)) - self.hp.max_abs_value,
                    -self.hp.max_abs_value, self.hp.max_abs_value)
            else:
                return np.clip(self.hp.max_abs_value * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)), 0, self.hp.max_abs_value)

        assert S.max() <= 0 and S.min() - self.hp.min_level_db >= 0
        if self.hp.symmetric_mels:
            return (2 * self.hp.max_abs_value) * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)) - self.hp.max_abs_value
        else:
            return self.hp.max_abs_value * ((S - self.hp.min_level_db) / (-self.hp.min_level_db))

    def _denormalize(self,D):
        if self.hp.allow_clipping_in_normalization:
            if self.hp.symmetric_mels:
                return (((np.clip(D, -self.hp.max_abs_value,
                                  self.hp.max_abs_value) + self.hp.max_abs_value) * -self.hp.min_level_db / (2 * self.hp.max_abs_value))
                        + self.hp.min_level_db)
            else:
                return ((np.clip(D, 0, self.hp.max_abs_value) * -self.hp.min_level_db / self.hp.max_abs_value) + self.hp.min_level_db)

        if self.hp.symmetric_mels:
            return (((D + self.hp.max_abs_value) * -self.hp.min_level_db / (2 * self.hp.max_abs_value)) + self.hp.min_level_db)
        else:
            return ((D * -self.hp.min_level_db / self.hp.max_abs_value) + self.hp.min_level_db)