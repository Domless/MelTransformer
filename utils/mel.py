
import torch
import librosa
import torchaudio.transforms as T

class MelTranform:

    @classmethod
    def from_h(cls, h, device) -> '''MelTranform''':
        return cls(
            device, 
            h.sampling_rate, 
            h.n_fft, 
            h.mel_hop_size,
            h.num_mels,
            h.win_size,
            h.fmin,
            h.fmax
        )

    def __init__(self, device, sr=16000, n_fft=1024, hop=256, n_mels=80, win_size=1024, f_min=100, f_max=8000, center=False):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, 
            n_fft=n_fft, 
            win_length=win_size, 
            hop_length=hop, 
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            normalized=False,
            onesided=True,
            pad_mode='reflect',
            center=center
        ).to(device)

        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmax=f_max
        ).T
        self.mel_transform.mel_scale.fb.copy_(torch.tensor(mel_basis))
        
    def prepare(self, data):
        # m = self.mel_transform(data)
        # m = m.clamp_(min=1e-5).log_()
        return torch.log(torch.clamp(self.mel_transform(data), min=1e-5))
        #return m