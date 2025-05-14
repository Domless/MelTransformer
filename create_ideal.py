import numpy as np
from utils.utils import shift, files_with_type_fiter, load_l0_vocals

f0_ideals = load_l0_vocals(
            ideals_path, self.framesamp, self.hop, self.voice_min_hz, self.voice_max_hz, sr=self.sr
)
np.save("d1.npy", f0_ideals)
d2=np.load("d1.npy")