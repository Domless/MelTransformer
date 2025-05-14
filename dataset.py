import os
import os.path as os_path
from tqdm import tqdm
import json
import random
import numpy
import torch
import torchaudio
import torchaudio.transforms as T
import os
from torch.utils.data import Dataset

from utils.frame import VocalData, freq_to_note, load_vocals
from utils.utils import shift, files_with_type_fiter, load_l0_vocals
from utils.mel import MelTranform
SETS = 2

# 🔹 Dataset для загрузки и обработки аудио
class AudioDataset(Dataset):

    @staticmethod
    def load_wavs(path, framesamp, min_hz, max_hz, sr):
        folders = [f for f in os.listdir(path) if os_path.isdir(os_path.join(path, f))]
        files_dict = {}
        for f in folders:
            wavs = files_with_type_fiter(os_path.join(path, f), ".wav", False)
            for w in wavs:
                waveform, sample_rate = torchaudio.load(os_path.join(path, f, w))
                #print(waveform.max(), waveform.min())
                            # Ресэмплим в 16кГц, если нужно
                if sr != sample_rate:
                    resampler = T.Resample(orig_freq=sample_rate, new_freq=sr)
                    waveform = resampler(waveform)

                waveform = torchaudio.functional.lowpass_biquad(waveform, sr, max_hz)
                num_chunks = waveform.shape[1] // framesamp
                files_dict[f + "/" + w] = waveform[:, :num_chunks * framesamp].reshape(
                    1, num_chunks, framesamp
                ).squeeze(0)
                

        return files_dict
    
    def create_mel_ideals(self, ideals):
        mel_ideals = {}
        for k, vs in ideals.items():
            mel_ideals[k] = []
            for v in tqdm(vs, f"caching - {k}"):
                t = torch.Tensor(v).to(self.device)
                t_mel=self._mel_prepare(t)
                mel_ideals[k].append(t_mel.cpu())
                del t, t_mel
        return mel_ideals
    
    
    def create_cache_shift(self):
        self.cache = {}
        for i in tqdm(range(len(self.data) * SETS), "caching"):

            segment = self.data[i//SETS]

            f1_name = self.vocals_list[segment["f1"]]["name"]
            f2_name = self.vocals_list[segment["f2"]]["name"]
            pos = segment["frame_id"]
            frequencies_f1 = segment["f1_hz"]
            frequencies_f2 = segment["f2_hz"]

            if i % 2 == 1:
                f2_name, f1_name = f1_name, f2_name
                frequencies_f1, frequencies_f2 = frequencies_f2, frequencies_f1

            last_cur = self.wavs[f1_name][pos:pos+self.line_len, :].to(self.device)
            cur = shift(
                last_cur, frequencies_f1, frequencies_f2, self.sr, self.device
            ).reshape(1, -1).squeeze()
            self.cache[i] = self._mel_prepare(cur).cpu()
            del cur, last_cur
            torch.cuda.empty_cache()

    def _seg_to_mel(self, name, pos):
        return self._mel_prepare(self.wavs[name][pos:pos+self.line_len, :].reshape(1, -1).squeeze().to(self.device))
    
    def create_full_cache(self):
        cache = {}

        for idx in tqdm(range(len(self)), "caching dataset"):
            pos, (f1_name, _), (f2_name, _) = self.get_wavs_position(idx)

            if not cache.get(f1_name):
                cache[f1_name] = {}
            if pos not in cache[f1_name]:
                cache[f1_name][pos] = self._seg_to_mel(f1_name, pos).cpu()

            if not cache.get(f2_name):
                cache[f2_name] = {}
            if pos not in cache[f2_name]:
                cache[f2_name][pos] = self._seg_to_mel(f2_name, pos).cpu()

        return cache

    def __init__(
            self, 
            folder_path,
            ideals_path,
            device, 
            h,
            with_shift=False,
            use_cache=False,
            cahce_folder = "./cache"
        ):
        cache_ideals_file = f"{cahce_folder}/ideals.npy"
        cache_dataset_file = f"{cahce_folder}/data.npy"

        self.device = device
        self.with_shift = with_shift
        self._use_cache = use_cache

        self.sr=h.sampling_rate
        self.n_fft = h.n_fft
        self.hop = h.hop_size
        self.framesamp = h.segment_size
        self.min_hz = h.fmin
        self.max_hz = h.fmax
        self.n_mels = h.num_mels
        self.win_size = h.win_size
        self.voice_min_hz = h.voice_fmin
        self.voice_max_hz = h.voice_fmax
        self.mel_hop_size = h.mel_hop_size

        random.seed(h.seed)
        self.folder = folder_path
        with open(folder_path + '/train.json', 'r') as file:
            self.full_data = json.load(file)
            self.line_len = self.full_data["line_len"]
            self.vocals_list = self.full_data["vocals_list"]
            self.data = self.full_data["data"]

        self.mel_transform = MelTranform(
            device, self.sr, self.n_fft, self.mel_hop_size, self.n_mels, self.win_size, self.min_hz, self.max_hz
        )

        if self._use_cache:
            if os_path.exists(cache_dataset_file):
                self.cache_dataset = numpy.load(cache_dataset_file, allow_pickle=True).flat[0]
            else:
                self.wavs = self.load_wavs(
                    folder_path + "/wav", self.framesamp, self.voice_min_hz, self.voice_max_hz, self.sr
                )
                self.cache_dataset = self.create_full_cache()
                numpy.save(cache_dataset_file, self.cache_dataset)
                self.wavs = None
                #del self.wavs
        else:
            self.wavs = self.load_wavs(
                folder_path + "/wav", self.framesamp, self.voice_min_hz, self.voice_max_hz, self.sr
            )

        # if self.with_shift:
        #     self.create_cache_shift()
        if os_path.exists(cache_ideals_file):
            self.f0_ideals = numpy.load(cache_ideals_file, allow_pickle=True).flat[0]
        else:
            self.f0_ideals = load_l0_vocals(
                ideals_path, self.framesamp, self.hop, self.voice_min_hz, self.voice_max_hz, sr=self.sr
            )
            numpy.save(cache_ideals_file, self.f0_ideals)
        
        self.mel_ideal_cache = None
        if self._use_cache:
            self.mel_ideal_cache = self.create_mel_ideals(self.f0_ideals)
            self.f0_ideals = None

        print("dsdsd")

    def __len__(self):
        return len(self.data) * SETS
    
    def normalize_per_sequence(self, magnitude):
        return magnitude / (magnitude.max(dim=-1, keepdim=True)[0] + 1e-6)
    
    def _mel_prepare(self, data):
        return self.mel_transform.prepare(data)
    
    def get_wavs_position(self, idx):
        segment = self.data[idx//SETS]
        return (
            segment["frame_id"], 
            (self.vocals_list[segment["f1"]]["name"], segment["f1_hz"]), 
            (self.vocals_list[segment["f2"]]["name"], segment["f2_hz"])
        )
    
    def __getitem__(self, idx):
        if self._use_cache:
            return self.__getitem__cache(idx)
        return self.__getitem__no_cache(idx)

    # NO CACHE
    def __getitem__no_cache(self, idx):
        pos, (f1_name, frequencies_f1), (f2_name, frequencies_f2) = self.get_wavs_position(idx)

        if idx % 2 == 1:
            f2_name, f1_name = f1_name, f2_name
            frequencies_f1, frequencies_f2 = frequencies_f2, frequencies_f1

        #ideal = random.choice(self.mel_ideal_cache[freq_to_note(frequencies_f2[0])]).to(self.device)
        ideal = self._mel_prepare(torch.Tensor(
                [random.choice(self.f0_ideals[freq_to_note(f)]) for f in frequencies_f2]
            ).reshape(1, -1).squeeze().to(self.device))

        dist = self._seg_to_mel(f2_name, pos)
        # if self.with_shift:
        #     cur_mel = self.cache[idx].to(self.device)
        #     return cur_mel, dist, ideal
        
        cur = self._seg_to_mel(f1_name, pos)
        return cur, dist, ideal, torch.tensor(frequencies_f2[0]).to(self.device)


    # CACHE
    def __getitem__cache(self, idx):
        pos, (f1_name, frequencies_f1), (f2_name, frequencies_f2) = self.get_wavs_position(idx)

        if idx % 2 == 1:
            f2_name, f1_name = f1_name, f2_name
            frequencies_f1, frequencies_f2 = frequencies_f2, frequencies_f1

        f1 = self.cache_dataset[f1_name][pos].to(self.device)
        f2 = self.cache_dataset[f2_name][pos].to(self.device)

        ideal = random.choice(self.mel_ideal_cache[freq_to_note(frequencies_f2[0])]).to(self.device)
        
        return f1, f2, ideal, torch.tensor(frequencies_f2[0]).to(self.device)
