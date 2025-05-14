import math
import random
import scipy.signal as signal
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import crepe

SILENCE = 100
NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

def is_silence(x):
    if np.abs(x).mean() < SILENCE:
        return True
    return False
    
def trim(data, frames = 1024, hop = 1024):
    return np.array(
        [(data[i : i + frames]) for i in range(0, len(data) - frames, hop)]
    )

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

# -20 в фломуле так-то 69, а не 49
def note_number_to_frequency(note_number: int) -> float:
    """Переводит номер ноты в частоту (Гц)."""
    return 440 * (2 ** ((note_number - 49) / 12))

def freq_to_note(freq):
    if freq == 0:
        return None
    note_number = 12 * math.log2(freq / 440) + 49
    return round(note_number)

class Frame:
    def __init__(self, data, position, sample_rate, prev_data=None) -> None:
        self.data = data
        self.prev_data = prev_data
        self.position = position
        self.sample_rate = sample_rate
        self.freq = None
        self.confidence = None
        self._note = None
        self.training_freq = None

    def detect(self, detector, idx):
        self.freq = (
            0 if is_silence(self.data) else detector.detect(self.data, idx)
        )
        self._note = freq_to_note(self.freq)
        return self.freq

    @property
    def note(self) -> tuple[int, str] | None:
        if self._note:
            return (
                int((self._note + 8) // len(NOTES)),
                NOTES[(self._note - 1) % len(NOTES)],
            )
        return None


class ParentData:
    def __init__(
        self, file_name, framesamp=1024, hop=1024, min_hz=10, max_hz=600, model_rate=16000, index=None, vocal=None, vocal_rate=44100
    ) -> None:
        self.index = index
        self.file_name = file_name
        self.file_name_only = Path(file_name).name
        self.framesamp = framesamp
        self.hop = hop
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.frames: list[Frame] = []

        if vocal is None:
            self.sample_rate, self.raw_data = wav.read(self.file_name)
        else:
            self.sample_rate = vocal_rate
            self.raw_data = vocal
        # resample audio if necessary
        if self.sample_rate != model_rate:
            from resampy import resample
            self.raw_data = resample(self.raw_data, self.sample_rate, model_rate)
            self.sample_rate = model_rate


    def load(self):
        step_size = self.framesamp/self.sample_rate*1000
        filtered = bandpass(
            self.raw_data, [self.min_hz, self.max_hz], self.sample_rate
        )
        data = trim(filtered, self.framesamp, self.hop)
        predetect = crepe.predict(
            self.raw_data, self.sample_rate, viterbi=True, step_size=step_size
        )
        time, frequency, confidence, activation = predetect
        binary_confidence = np.where(confidence > 0.5, 1, 0)
        coef = self.sample_rate / (1000 / step_size)
        for idx, d in enumerate(data):
            pre_X = None if idx == 0 else data[idx - 1]
            frame = Frame(d, idx * self.hop, self.sample_rate, pre_X)
            # left_index = math.ceil((idx * self.hop) / coef)
            # right_index = int((idx * self.hop + self.framesamp - 1) / coef) + 1
            # frame.freq = frequency[left_index:right_index].mean()
            frame.freq = frequency[idx]
            frame._note = freq_to_note(frame.freq)
            frame.confidence = confidence[idx]
            self.frames.append(frame)
            # if left_index == right_index:
            #     print("EQQQQ")
        self.create_filtred_data_by_note2()
        return self.frames

    def create_filtred_data(self, neib_size=7, border=11, dif=0.1, con=0.6):
        for i in range(neib_size):
            self.frames[i].training_freq = None
            self.frames[len(self.frames) - i - 1].training_freq = None

        for idx in range(neib_size, len(self.frames) - neib_size):
            self.frames[idx].training_freq = None
            cur_value = self.frames[idx].freq
            confidence = self.frames[idx].confidence
            if cur_value == 0 or cur_value > self.max_hz or cur_value < self.min_hz or confidence < con:
                continue
            naib_rate = 0
            for nidx in range(idx - neib_size, idx + neib_size + 1):
                neib_value = self.frames[nidx].freq
                neib_confidence = self.frames[nidx].confidence
                if neib_value == 0 or idx == nidx or neib_confidence < con:
                    continue
                if abs(1 - (neib_value / cur_value)) < dif:
                    naib_rate += 1

            if naib_rate >= border:
                self.frames[idx].training_freq = cur_value


    def create_filtred_data_by_note(self, neib_size=2, con=0.5, min_delta = 0.1):
        min_ = self.min_hz - min_delta * self.min_hz 
        for i in range(neib_size):
            self.frames[i].training_freq = None
            self.frames[len(self.frames) - i - 1].training_freq = None

        for idx in range(neib_size, len(self.frames) - neib_size):
            self.frames[idx].training_freq = None
            frame = self.frames[idx]
            if frame.freq == 0 or frame.confidence < con or frame.freq > self.max_hz or frame.freq < min_:
                continue
            
            is_line = True
            for nidx in range(idx - neib_size, idx + neib_size + 1):
                neib = self.frames[nidx]
                if idx == nidx:
                    continue

                if neib.freq == 0 or neib.confidence < con or abs(neib._note-frame._note) > 1:
                    is_line = False
                    break

            if is_line:
                self.frames[idx].training_freq = frame.freq


    #def create_filtred_data_by_note2(self, neib_size=1, con=0.75, min_delta = 0.1, note_delta = -0.2):
    #def create_filtred_data_by_note2(self, neib_size=2, con=0.70, min_delta = 0.1, note_delta = 0.25):
    def create_filtred_data_by_note2(self, neib_size=1, con=0.51, min_delta = 0.1, note_delta = 0.5):
        min_ = self.min_hz - min_delta * self.min_hz 
        for i in range(neib_size):
            self.frames[i].training_freq = None
            self.frames[len(self.frames) - i - 1].training_freq = None

        for idx in range(neib_size, len(self.frames) - neib_size):
            self.frames[idx].training_freq = None
            frame = self.frames[idx]
            if frame.freq == 0 or frame.confidence < con or frame.freq > self.max_hz or frame.freq < min_:
                continue
            
            is_line = True
            delta = abs(note_number_to_frequency(frame._note) - note_number_to_frequency(frame._note + 1))
            delta = delta + delta * note_delta
            for nidx in range(idx - neib_size, idx + neib_size + 1):
                neib = self.frames[nidx]
                if idx == nidx:
                    continue

                if neib.freq == 0 or neib.confidence < con or abs(neib.freq-frame.freq) > delta:
                    is_line = False
                    break

            if is_line:
                self.frames[idx].training_freq = frame.freq


class NoteData(ParentData):

    def __init__(
        self, file_name, framesamp=1024, hop=1024, min_hz=10, max_hz=600
    ) -> None:
        super().__init__(file_name, framesamp, hop, min_hz, max_hz)
        self.frames_by_notes: dict = {}

    def prepare_frames(self):
        for f in self.frames:
            if f._note not in self.frames_by_notes:
                self.frames_by_notes[f._note] = [f]
            else:
                self.frames_by_notes[f._note].append(f)

    def get_frames_by_note(self, note: int):
        if not self.frames_by_notes:
            self.prepare_frames()
        return self.frames_by_notes.get(note)

    def get_randon_frame_by_note(self, note: int):
        frames = self.get_frames_by_note(note)
        if not frames:
            return
        return frames[random.randint(len(frames))]


class VocalData(ParentData):
    def get_freqs(self):
        return [frame.freq for frame in self.frames]

    def get_filtred_data(self):
        return [
            np.inf if frame.training_freq is None else frame.freq
            for frame in self.frames
        ]


def load_vocals(
    files: list[str], framesamp=1024, hop=1024, min_hz=10, max_hz=500, from_index=None, sr=16000
) -> list[VocalData]:
    vocals = []
    for idx, f in enumerate(files, from_index if from_index is not None else 0):
        vocal = VocalData(
            f, framesamp, hop, min_hz, max_hz, model_rate=sr, index=idx if from_index is not None else None
        )
        vocal.load()
        # f = vocal.get_filtred_data()
        vocals.append(vocal)
    return vocals


def make_plot(vocals: list[VocalData]):
    colors = ["black", "orange", "green", "red", "blue", "y"]
    for idx, c in enumerate(vocals):
        freqs = c.get_freqs()
        plt.plot(np.arange(len(freqs)), freqs, color=colors[idx % len(colors)], label=c.file_name, linestyle ='dashed')
        filtred = c.get_filtred_data()
        plt.plot(
            np.arange(len(freqs)),
            filtred,
            color=colors[idx % len(colors)],
            label=c.file_name,
        )
    plt.legend(loc="upper right")
    plt.show()
