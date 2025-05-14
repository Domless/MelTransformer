import glob
import json
import os
import os.path as os_path
import torch
import numpy as np
import torchaudio.transforms as T
import librosa
import torchaudio
import torchaudio.transforms as T
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils.frame import VocalData, load_vocals, freq_to_note

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config(config_file):
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    return AttrDict(json_config)

def plot_spectrograms(wave, new_wave, title_wave="tr"):
    plt.figure(figsize=(10, 5))  # Создаём одно окно для обоих графиков

    # Первая спектрограмма
    plt.subplot(2, 1, 1)
    plt.imshow(wave, aspect='auto', origin='lower')
    plt.colorbar(label="Амплитуда")
    plt.xlabel("Фреймы (время)")
    plt.ylabel("Мел-частоты")
    plt.title(f"Оригинальная мел-спектрограмма ({title_wave})")

    # Вторая спектрограмма
    plt.subplot(2, 1, 2)
    plt.imshow(new_wave, aspect='auto', origin='lower')
    plt.colorbar(label="Амплитуда")
    plt.xlabel("Фреймы (время)")
    plt.ylabel("Мел-частоты")
    plt.title("Обработанная мел-спектрограмма")

    plt.tight_layout()  # Автоматическая подгонка размеров
    plt.show()


def plot_spectrograms__(waves, labels):
    plt.figure(figsize=(10, 5))  # Создаём одно окно для обоих графиков

    # Первая спектрограмма
    for idx, w in enumerate(waves,):
        plt.subplot(len(waves), 1, idx+1)
        plt.imshow(w, aspect='auto', origin='lower')
        plt.colorbar(label="Амплитуда")
        plt.xlabel("Фреймы (время)")
        plt.ylabel("Мел-частоты")
        plt.title(labels[idx])

    plt.tight_layout()  # Автоматическая подгонка размеров
    plt.show()


    # 🔹 Функция для отрисовки спектрограмм
def plot_spectrogram(mel_spec, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.log2().detach().cpu().numpy(), aspect="auto", origin="lower")
    plt.colorbar(label="Log Amplitude")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bands")
    plt.show()

# 🔹 Функция для отрисовки waveforms
def plot_waveform(waveform1, title, sample_rate=16000):
    plt.figure(figsize=(10, 3))
    time_axis = np.linspace(0, len(waveform1) / sample_rate, num=len(waveform1))
    plt.plot(time_axis, waveform1, linewidth=0.8, color="green")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# 🔹 Функция визуализации
def plot_waveforms(orig_wave, new_wave):
    plt.figure(figsize=(12, 6))

    # Оригинальный и восстановленный сигнал
    plt.plot(orig_wave, label="Оригинал", color="green")
    plt.plot(new_wave, label="Восстановленный", color="red", alpha=0.7)
    plt.title("Временной сигнал")
    plt.legend()

    plt.show()


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")

def graph(ds, colors = "orange"):
    if not isinstance(ds, list):
        ds = [ds]
    if not isinstance(colors, list):
        colors = [colors]
    for d, color in zip(ds, colors):
        plt.plot(np.arange(len(d)), d, color=color)
    plt.show()

def load_checkpoint(filepath, device):
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def prepare_and_save_checkpoints(checkpoint_path, generator, steps, epoch):
    save_checkpoint(
        "{}/tr_{:08d}".format(checkpoint_path, steps), 
        {'generator': generator.state_dict(), "steps":steps, "epoch": epoch}
    )

def files_with_type_fiter(path, type, need_join=True):
        return [
            os_path.join(path, f) if need_join else f
            for f in os.listdir(path)
            if os_path.isfile(os_path.join(path, f)) and f.endswith(type)
        ]

# export ROCM_PATH=/opt/rocm
def load_wavs_f0(ideals: list[VocalData], framesamp, max_hz, sr=16000):
    files_dict = {}
    for w in ideals:
        raw_data, sample_rate = torchaudio.load(w.file_name)
                            # Ресэмплим в 16кГц, если нужно
        if sr != sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=sr)
            waveform = resampler(raw_data)

        waveform = torchaudio.functional.lowpass_biquad(waveform, sr, max_hz)
        num_chunks = waveform.shape[1] // framesamp
        files_dict[w.file_name] = waveform[:, :num_chunks * framesamp].reshape(
                num_chunks, framesamp
        ).squeeze(0)

    return files_dict


def load_l0_vocals(path, frames, hop, min_hz, max_hz, sr=16000):
        #files = ["./data/ideals/1.wav", "./data/ideals/2.wav", "./data/ideals/3.wav"]
        #files = ["./data/ideal2.wav", "./data/c_ideal2.wav"]

    files = files_with_type_fiter(path, ".wav", True)
    ideals = load_vocals(files, frames, hop, min_hz, max_hz, sr=sr)
    torch_files = load_wavs_f0(ideals, frames, max_hz, sr)


    min_note, max_note = freq_to_note(min_hz), freq_to_note(max_hz)
    ideal_dict = {i: [] for i in range(min_note, max_note + 1)}

    for ideal in ideals:
        for idx, frame in enumerate(ideal.frames):
            if not frame.training_freq or ideal_dict.get(frame._note) is None:
                continue
            ideal_dict[frame._note].append(torch_files[ideal.file_name].data[idx].tolist())

    for key, item in ideal_dict.items():
        if not item:
            print(f"{key} - не имеет данных")

    return ideal_dict


def detect_f0(file, framesamp, hop, min_hz, max_hz, sr, n_steps):
    import scipy.io.wavfile as wav
    sample_rate, raw_data = wav.read(file)
    from resampy import resample
    raw_data = resample(raw_data, sample_rate, sr)

    y_shifted = librosa.effects.pitch_shift(raw_data, sr=sample_rate, n_steps=n_steps)
    vocal = VocalData(
        file, framesamp, hop, min_hz, max_hz, model_rate=sr, vocal=y_shifted, vocal_rate=sr
    )
    vocal.load()
    return vocal.get_freqs()


pitch_shift = None
def shift(data, from_freq, to_freq, sr, device):
    global pitch_shift
    if not pitch_shift:
        pitch_shift = T.PitchShift(sr, 1).to(device)
    # Применение сдвига по высоте для каждого блока
    for i in range(len(from_freq)):
        # Рассчитываем разницу частот и преобразуем в количество полутонов
        n_steps = 12 * np.log2(to_freq[i] / from_freq[i])
        pitch_shift.n_steps = round(n_steps)
        # Создаем преобразование PitchShift
        #pitch_shift = T.PitchShift(sr, n_steps=round(n_steps.item())).to(device)  # Используем .item() для преобразования в обычное число
        # Используем torch.no_grad(), чтобы избежать вычисления градиентов для операций преобразования
        with torch.no_grad():
            # Применяем PitchShift к сигналу, чтобы избежать изменений inplace
            data[i] = pitch_shift(data[i].clone())  # clone() предотвращает inplace-изменения
    
    return data

def shift_by_steps(data, steps, sr, device):
    pitch_shift = T.PitchShift(sr, 1).to(device)
    # Применение сдвига по высоте для каждого блока
    for i in range(len(steps)):
        # Рассчитываем разницу частот и преобразуем в количество полутонов
        n_steps = steps[i]
        pitch_shift.n_steps = round(n_steps)
        # Создаем преобразование PitchShift
        #pitch_shift = T.PitchShift(sr, n_steps=round(n_steps.item())).to(device)  # Используем .item() для преобразования в обычное число
        # Используем torch.no_grad(), чтобы избежать вычисления градиентов для операций преобразования
        with torch.no_grad():
            # Применяем PitchShift к сигналу, чтобы избежать изменений inplace
            data[i] = pitch_shift(data[i].clone())  # clone() предотвращает inplace-изменения
    
    return data

def shift_full(data, step, sr, device):
    pitch_shift = T.PitchShift(sr, step).to(device)
    with torch.no_grad():
        return pitch_shift(data)