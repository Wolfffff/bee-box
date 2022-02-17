# %%

import logging
import pandas as pd
import utils.trx_utils as trx_utils
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis_logger")


# Load bee tracks for the first hour
bee_tracks_df = pd.read_csv(
    "/Genomics/ayroleslab2/scott/bees/bee-box/data/001/001_aruco_data_with_track_numbers.csv"
)

# %%
# Remove where tag is -1 and pivot to long format
bee_tracks_df = bee_tracks_df[["Frame", "Tag", "headX", "headY"]]

bee_tracks_df_x = bee_tracks_df[bee_tracks_df.Tag != -1].pivot(
    index="Frame", columns="Tag", values="headX"
)
bee_tracks_df_y = bee_tracks_df[bee_tracks_df.Tag != -1].pivot(
    index="Frame", columns="Tag", values="headY"
)

bee_ids = bee_tracks_df_x.columns.to_numpy()
# %%
# (frames, coordinates, bees)
bee_tracks = np.stack((bee_tracks_df_x.to_numpy(), bee_tracks_df_y.to_numpy()), axis=1)
bee_tracks = bee_tracks[:, np.newaxis, :, :]
frame_count, node_count, coords_count, instance_count = bee_tracks.shape
# bee_tracks[:, 1, :] = -bee_tracks[:, 1, :]

# %%
# Remove where there are too many missing values
missing_ct = np.count_nonzero(np.isnan(bee_tracks[:,:, 0, :]), axis=0)
missing_freq = missing_ct / bee_tracks.shape[0]
missingness_threshold = 0.2
missing_selector = (missing_freq < missingness_threshold).flatten()
bee_tracks = bee_tracks[:, :, :, missing_selector]
# Make sure we filter the ids too to keep everything in sync!
bee_ids = bee_ids[missing_selector]

# %%
# Ordering really matters on the fill pipeline
bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear", limit=20)

# Mask out "super fast" speed
px_mm = 15.6
fps = 20
threshold = 20 * (1 / px_mm) * (fps)
bee_vel = trx_utils.instance_node_velocities(bee_tracks, 0, bee_tracks.shape[0])
mask = np.stack(((bee_vel > threshold), (bee_vel > threshold)), axis=2)
mask.shape
bee_tracks[mask] = np.nan

# bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear", limit=10)
bee_tracks = trx_utils.smooth_gaussian(bee_tracks, std=2, window=10)
# bee_tracks = trx_utils.smooth_ewma(bee_tracks,alpha=0.3)


# %%
key_tracks_dict = {"keys": bee_ids, "tracks":bee_tracks}

import pickle
with open('tracks.pickle', 'wb') as handle:
    pickle.dump(key_tracks_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
with open('tracks.pickle', 'rb') as handle:
    key_tracks_dict = pickle.load(handle)

# %%

bee_tracks = key_tracks_dict["tracks"]
bee_ids = key_tracks_dict["keys"]

#%%

np.apply_along_axis(np.mean, 0, np.isnan(bee_tracks))[0,0,:]

# %%
