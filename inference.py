import glob
import os
import os.path as os_path
import random
import numpy
import json
import torch
import torchaudio
import torchaudio.transforms as T
#from scipy.io.wavfile import write
from models import MelTransformer2
from modules.style_encoder import StyleEncoder

from utils.frame import VocalData, freq_to_note, load_vocals
from utils.utils import shift, files_with_type_fiter, load_l0_vocals
from utils.mel import MelTranform

from dataset import AudioDataset

from utils.utils import (
    plot_spectrograms,
    scan_checkpoint,
    load_checkpoint, 
    prepare_and_save_checkpoints, 
    get_config, 
    plot_spectrograms__, 
    plot_waveforms,
    AttrDict
)
from utils.pitch import f0_to_coarse

from modules.hifigan import Generator as HGenerator

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

def save_wav():
    pass

def load_hifi_gan(checkpoint_file, device):
    print('Initializing Inference Process..')

    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    # torch.manual_seed(h.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(h.seed)
    generator = HGenerator(h).to(device)

    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    return generator


def inference(h, wav_path, target: float, hifi_gan_checkpoint, g_checkpoint, se_checkpoint, ideals_path, ideals_file = None):

    tr_generator = MelTransformer2(
        hidden_dim=h.tr_hidden_dim, num_layers=h.tr_num_layers, nhead=h.tr_nhead, ideal_dim=h.style_dim
    ).to(device)

    style_encoder = StyleEncoder(dim_in=h.dim_in, style_dim=h.style_dim, max_conv_dim=h.hidden_dim).to(device)
    pitch_embed = torch.nn.Embedding(300, h.style_dim, padding_idx=0).to(device)


    hifi_gan = load_hifi_gan(hifi_gan_checkpoint, device)

    mel_tranform = MelTranform.from_h(h, device)

    state_dict_g = load_checkpoint(g_checkpoint, device)
    tr_generator.load_state_dict(state_dict_g['generator'])

    state_dict_se = load_checkpoint(se_checkpoint, device)
    style_encoder.load_state_dict(state_dict_se['encoder'])

    tr_generator.eval()
    style_encoder.eval()
    hifi_gan.eval()
    hifi_gan.remove_weight_norm()

    chunk_size = h.segment_size

    with torch.no_grad():

        waveform, file_sr = torchaudio.load(wav_path)
        if h.sampling_rate != file_sr:
            resampler = T.Resample(orig_freq=file_sr, new_freq=h.sampling_rate)
            waveform = resampler(waveform)
        waveform = torchaudio.functional.lowpass_biquad(waveform, h.sampling_rate, 1500)#[:, 8192*10:8192*15]
        print(waveform.shape)
        if os_path.exists(ideals_file):
            f0_ideals = numpy.load(ideals_file, allow_pickle=True).flat[0]
        else:
            f0_ideals = load_l0_vocals(
                ideals_path, chunk_size, h.hop_size, h.voice_fmin, h.voice_fmax, sr=h.sampling_rate
            )
        target_tensors = pitch_embed(f0_to_coarse(torch.tensor(target).to(device)))
        note = freq_to_note(target)

        #num_chunks = waveform.shape[1] // chunk_size
        #chunks = waveform[:, :num_chunks * chunk_size].reshape(1, num_chunks, chunk_size).squeeze(0).to(device)

        ideal = torch.Tensor(random.choice(f0_ideals[note])).to(device)
        ideal = mel_tranform.prepare(ideal).unsqueeze(0).to(device)
        ideal = style_encoder(ideal)
        dec_inp = torch.stack([target_tensors.unsqueeze(0), ideal.squeeze(1)], 1)
        

        #x = mel_tranform.prepare(chunks).permute(0, 2, 1).to(device)


        mel_orig = mel_tranform.prepare(waveform.to(device)).permute(0, 2, 1).to(device)
        num_chunks = mel_orig.shape[1] // 120
        mel_chunks = mel_orig[:, :num_chunks * 120, :].reshape(num_chunks, 120, 80).to(device)
        
        #print(x.shape)
        dec_inp = dec_inp.expand(mel_orig.shape[0], -1, -1)
        y_mel = tr_generator(mel_orig, dec_inp)

        # y_mel = y_mel + abs(y_mel.min())
        # y_mel = y_mel * (mel_orig.max() + 11.5129)/y_mel.max() - 11.5129

        # dec_inp = dec_inp.expand(mel_chunks.shape[0], -1, -1)
        # y_mel = tr_generator(mel_chunks, dec_inp)

        # min_vals = y_mel.min(dim=1, keepdim=True)[0]  # shape: [B, 1]
        # y_mel = y_mel + min_vals.abs()

        # y_max = mel_chunks.max(dim=1, keepdim=True)[0]  # shape: [B, 1]
        # y_g_hat_max = y_mel.max(dim=1, keepdim=True)[0]  # shape: [B, 1]

        # y_mel = y_mel * (y_max + 11.5129) / y_g_hat_max - 11.5129

        # # Склеим по временной оси
        mel_reshaped = y_mel.reshape(-1, 80).unsqueeze(0)  # [chunks * 40, 80]
        #mel_reshaped = y_mel
        #mel_reshaped = x.reshape(-1, 80).unsqueeze(0)  # [chunks * 40, 80]
        #mel_reshaped = x
        plot_spectrograms__(
            [
                mel_orig.squeeze(0).permute(1, 0).detach().cpu().numpy(), 
                mel_reshaped.squeeze(0).permute(1, 0).detach().cpu().numpy(), 
            ], 
            ["x", "y"]
        )
        print(mel_reshaped.shape)

        # # Теперь разобьём на отрезки длиной 33
        # total_frames = mel_reshaped.shape[0]
        # n_chunks = total_frames // 33

        # mel_chunks = mel_reshaped[:n_chunks * 33]  # Обрезаем лишнее, если есть
        # mel_chunks = mel_chunks.view(n_chunks, 33, 80).permute(0, 2, 1)  # [n_chunks, 80, 33]

        y_res = hifi_gan(y_mel.permute(0, 2, 1)) * 2#[:, :, :chunks.shape[1]]

        audio_s = y_res.squeeze().reshape(1, -1)
        audio_s = torchaudio.functional.lowpass_biquad(audio_s, h.sampling_rate, 1500)
        # audio = audio_s.squeeze().cpu().numpy()
        # for idx, d in enumerate(x):
        #     plot_spectrograms(d.detach().cpu().numpy(), y_mel[idx].detach().cpu().numpy())
        plot_waveforms(waveform.squeeze().cpu().numpy(), audio_s.squeeze().cpu().numpy())
        print(waveform.squeeze().cpu().numpy().max())
        print(audio_s.squeeze().cpu().numpy().max())

        output_file = wav_path[:-4] +  '_generated_tr.wav'
        torchaudio.save(output_file, audio_s.cpu(), h.sampling_rate)

        print(output_file)


def main():
    #inference("./voice_embedding/checkpoints/tr_00000010", "./hifi/checkpoints_old/g_00000060", "data/test", "data/test_out_tr")
    h = get_config("./configs/v1.json")
    inference(
        h,
        "./../prepare/data/ideals/1.wav",
        180,
        "./checkpoints_hifi/g_02500000", 
        "./checkpoints_finetune/tr_00001135", 
        "./checkpoints_finetune/se_00001135", 
        "./../prepare/data/ideals_",
        "./cache/ideals.npy",
        )


if __name__ == '__main__':
    main()

