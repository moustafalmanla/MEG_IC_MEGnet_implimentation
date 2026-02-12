
"""
ICA Feature Extraction Pipeline for MEGNet Preparation
------------------------------------------------------
This script:
1. Loads raw MEG
2. Preprocesses lightly
3. Runs ICA
4. Extracts per-component:
   - 1D temporal signals
   - Spatial topographies
5. Generates plots:
   - Temporal trace
   - 2D spatial map (topomap)

No output folder required. Everything visualized directly.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# USER INPUT
# ===============================

RAW_FILE = Path("your_file_meg.fif")   # <-- change to your file
N_ICA = 30
LOW_FREQ = 1.0
HIGH_FREQ = 40.0

# ===============================
# LOAD RAW MEG
# ===============================

print("Loading MEG file...")
raw = mne.io.read_raw_fif(RAW_FILE, preload=True, verbose=False)

# Keep only MEG channels
raw.pick_types(meg=True, eeg=False, eog=False, ecg=False)

# ===============================
# BASIC CLEANING
# ===============================

print("Filtering...")
raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
raw.notch_filter(50, verbose=False)

# Optional: downsample for speed
raw.resample(200)

# ===============================
# RUN ICA
# ===============================

print("Running ICA decomposition...")
ica = mne.preprocessing.ICA(
    n_components=N_ICA,
    method="fastica",
    random_state=97,
    max_iter="auto"
)

ica.fit(raw)

print("Extracting ICA sources...")
sources = ica.get_sources(raw).get_data()  # shape: (n_components, time)
mixing = ica.get_components()              # shape: (n_channels, n_components)

# ===============================
# VISUALIZATION LOOP
# ===============================

print("Generating IC plots...")

for ic in range(N_ICA):

    temporal = sources[ic]          # 1D temporal signal
    spatial = mixing[:, ic]         # spatial weights across sensors

    # -------- Plot temporal --------
    plt.figure(figsize=(10, 3))
    plt.plot(temporal, lw=0.7)
    plt.title(f"IC {ic} - Temporal Signal")
    plt.xlabel("Time points")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # -------- Plot spatial topography --------
    plt.figure(figsize=(4, 4))

    mne.viz.plot_topomap(
        spatial,
        raw.info,
        show=True,
        contours=0,
        sphere=0.09
    )

    plt.title(f"IC {ic} - Spatial Map")
    plt.show()

print("Done. You now have:")
print("- 1D temporal plots per IC")
print("- 2D spatial topography per IC")
print("These are the exact inputs required for MEGNet later.")

# ===============================
# OPTIONAL: SAVE ARRAYS
# ===============================

# np.save("ica_temporal.npy", sources)
# np.save("ica_spatial.npy", mixing)
