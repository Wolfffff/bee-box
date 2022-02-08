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
# bee_tracks[:, 1, :] = -bee_tracks[:, 1, :]

# %%
# Remove where there are too many missing values
missing_ct = np.count_nonzero(np.isnan(bee_tracks[:, 0, :]), axis=0)
missing_freq = missing_ct / bee_tracks.shape[0]
missingness_threshold = 0.2
bee_tracks = bee_tracks[:, :, missing_freq < missingness_threshold]

# Make sure we filter the ids too to keep everything in sync!
bee_ids = bee_ids[missing_freq < missingness_threshold]

# %%
# Ordering really matters on the fill pipeline
frame_count, node_count, instance_count = bee_tracks.shape
bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear",limit=10)

# Mask out "super fast" speed
px_mm = 13.6
fps = 20
threshold = 50 * (1/px_mm) * (fps)
bee_vel = trx_utils.instance_node_velocities_bees(bee_tracks, 0, bee_tracks.shape[0])
mask = np.stack(((bee_vel > threshold), (bee_vel > threshold)), axis=1)
bee_tracks[mask] = np.nan

# bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear", limit=10)
# bee_tracks = trx_utils.smooth_median(bee_tracks, window=3)
# bee_tracks = trx_utils.smooth_ewma(bee_tracks,alpha=0.3)


# %%
# Load videos
import importlib
importlib.reload(trx_utils)
bee_tracks_expanded = bee_tracks[:,np.newaxis,:,:]
video = "/Genomics/ayroleslab2/scott/bees/bee-box/data/20210923_run001_001.mp4"
# trx_utils.plot_trx(bee_tracks_expanded, video,frame_start=200, shift = -10, frame_end=220)


# %%

nonfocal_tag_list = [tag for tag in bee_tracks_df.Tag.unique()]
color_map = {nonfocal_tag_list[i]: "blue" for i in range(len(nonfocal_tag_list))}
color_map[21] = "red"

# %%
g1 = set({0, 1, 5, 8, 12, 19, 34, 35, 38, 40, 44, 47, 48, 49, 51, 55, 58, 60, 65, 70, 73, 74, 75})
g2 = set({2, 6, 10, 13, 14, 17, 18, 20, 22, 24, 27, 30, 42, 52, 53, 54, 56, 61, 62, 71})

for i in g1:
    color_map[i] = "red"
for i in g2:
    color_map[i] = "blue"
# %%

import importlib
importlib.reload(trx_utils)
trx_utils.plot_trx(bee_tracks_expanded, video,frame_start=0,trail_length=10, shift = 10, frame_end=1*10*20,color_map = color_map,id_map = bee_ids,scale_factor=(3672/3600),output_path="20210208_3s_cleaner")

# %%
# from sklearn.metrics.pairwise import nan_euclidean_distances
# import skvideo
# output_path="interaction"
# ffmpeg_writer = skvideo.io.FFmpegWriter(
#     f"{output_path}_locations.mp4", outputdict={"-vcodec": "libx264","-vcodec": "libx264"}
# )
# import cv2
# cap = cv2.VideoCapture(video)
# frame_start = 0
# frame_end = 10
# shift = 10
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + shift - 1)
# data = bee_tracks[frame_start:frame_end, :, :]
# dpi = 300


# # %%
# import palettable
# id_map = bee_ids
# color_map = {id_map[i]: "blue" for i in range(len(id_map))}
# # %%
# for frame_idx in range(data.shape[0]):
#     print(f"Frame {frame_idx}")
#     data_subset = data[frame_idx, :, :]
#     distances = nan_euclidean_distances(data_subset, data_subset)
#     for idx in range(2, data_subset.shape[0]):
#     # Note that you need to use single steps or the data has "steps"

#     # color = color_map[id_map[fly_idx]]
#     plt.plot(
#         data_subset[(idx - 2) : idx,0 , fly_idx] * scale_factor,
#         data_subset[(idx - 2) : idx, 1, fly_idx] * scale_factor,
#         linewidth=3 * idx / data_subset.shape[0]
#     )
# %%
