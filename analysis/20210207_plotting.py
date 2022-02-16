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
bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear", limit=20)

# Mask out "super fast" speed
px_mm = 15.6
fps = 20
threshold = 20 * (1 / px_mm) * (fps)
bee_vel = trx_utils.instance_node_velocities_bees(bee_tracks, 0, bee_tracks.shape[0])
mask = np.stack(((bee_vel > threshold), (bee_vel > threshold)), axis=1)
bee_tracks[mask] = np.nan

# bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear", limit=10)
bee_tracks = trx_utils.smooth_gaussian(bee_tracks, std=2, window=10)
# bee_tracks = trx_utils.smooth_ewma(bee_tracks,alpha=0.3)


# %%
# Load videos
import importlib

importlib.reload(trx_utils)
bee_tracks_expanded = bee_tracks[:, np.newaxis, :, :]
video = "/Genomics/ayroleslab2/scott/bees/bee-box/data/20210923_run001_001.mp4"
# trx_utils.plot_trx(bee_tracks_expanded, video,frame_start=200, shift = -10, frame_end=220)


# %%

nonfocal_tag_list = [tag for tag in bee_tracks_df.Tag.unique()]
color_map = {nonfocal_tag_list[i]: "#C8F2EF" for i in range(len(nonfocal_tag_list))}
color_map[21] = "#BB6464"

# %%
g1 = set(
    {
        0,
        1,
        5,
        8,
        12,
        19,
        34,
        35,
        38,
        40,
        44,
        47,
        48,
        49,
        51,
        55,
        58,
        60,
        65,
        70,
        73,
        74,
        75,
    }
)

g2 = set({2, 6, 10, 13, 14, 17, 18, 20, 22, 24, 27, 30, 42, 52, 53, 54, 56, 61, 62, 71})

for i in g1:
    color_map[i] = "#BB6464"
for i in g2:
    color_map[i] = "#C8F2EF"
# %%

import importlib
importlib.reload(trx_utils)
# bee_tracks_expanded[:, :, 1, :] = -bee_tracks_expanded[:, :, 1, :]
# trx_utils.plot_trx(
#     bee_tracks_expanded,
#     video_path=video,  # video,
#     frame_start=0,
#     trail_length=20,
#     shift=10,
#     frame_end=1 * 60 * 20,
#     color_map=color_map,
#     id_map=bee_ids,
#     scale_factor=(3672 / 3600),
#     output_path="video_anno.mp4",
#     annotate=True,
# )

# %%
import cv2
from matplotlib.collections import LineCollection
import matplotlib as mpl
id_map=bee_ids
color_map=color_map
mpl.rcParams['lines.solid_capstyle'] = 'round'
mpl.rcParams['lines.solid_joinstyle'] = 'round'
angles=np.repeat(0,999999)
video_path = video
frame_start = 0
frame_end = 1*30*fps
trail_length=20
import skvideo
import time
fly_ids=[7]
fly_id = 7
import palettable
ctr_idx = 0
expmt_name = "test"
tracks = bee_tracks_expanded[frame_start:frame_end, :, :, :]
output_path=f'14s_ego.mp4'#{time.strftime("%Y%m%d_%H%M%S")}_{frame_start}to{frame_end}_{expmt_name}_fly{fly_id}_raw_ego.mp4'
ffmpeg_writer = skvideo.io.FFmpegWriter(
    f"{output_path}", outputdict={"-vcodec": "libx264"}
)
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + 10 - 1)
dpi = 300
if tracks.ndim == 3:
    tracks = tracks[:, :, :, np.newaxis]
data = tracks[frame_start:frame_end, :, :, :]
print(data.shape)
for frame_idx in range(data.shape[0]):
    if cap.isOpened():
        res, frame = cap.read()

        # frame = frame[:,:,0]

        height, width, nbands = 96 * 5, 96 * 5, 3
        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)
        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        # Hide spines, ticks, etc.
        ax.axis("off")
        ax.imshow(frame, cmap="gray")
    else:
        fig, ax = plt.subplots()
    print(f"Frame {frame_idx}")
    data_subset = data[max((frame_idx - trail_length), 0) : frame_idx, :, :, :]
    for fly_idx in range(data_subset.shape[3]):
                for node_idx in range(data_subset.shape[1]):
                    for idx in range(2, data_subset.shape[0]):
                        # Note that you need to use single steps or the data has "steps"
                        if color_map == None:
                            plt.plot(
                                data_subset[(idx - 2) : idx, node_idx, 0, fly_idx]*(3672/3600),
                                data_subset[(idx - 2) : idx, node_idx, 1, fly_idx]*(3672/3600),
                                linewidth=6 * idx / data_subset.shape[0],
                                color=palettable.tableau.Tableau_20.mpl_colors[node_idx],
                            )
                        else:
                            color = color_map[id_map[fly_idx]]
                            (l,) = ax.plot(
                                data_subset[(idx - 2) : idx, node_idx, 0, fly_idx]*(3672/3600),
                                data_subset[(idx - 2) : idx, node_idx, 1, fly_idx]*(3672/3600),
                                linewidth=6 * idx / data_subset.shape[0],
                                color=color,
                            )
                            l.set_solid_capstyle("round")

    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # fig.set_size_inches(5, 5, True);
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.axis('off')
    # fig.patch.set_visible(False)
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)
    xy = data[frame_idx, ctr_idx, 0:2, fly_idx]
    x = xy[0]*(3672/3600)
    y = xy[1]*(3672/3600)
    ax.set_xlim((x - 320), (x + 320))
    ax.set_ylim((y - 320), (y + 320))

    image_from_plot = trx_utils.get_img_from_fig(fig, dpi)
    # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if angles is not None:
        # print(angles)
        image_from_plot = trx_utils.rotate_image(
            image_from_plot, -angles[frame_idx + frame_start] * 180 / np.pi
        )
        image_from_plot = image_from_plot[96:-96, 96:-96, :]
    ffmpeg_writer.writeFrame(image_from_plot)
    plt.close()
    # fig.close()
ffmpeg_writer.close()
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
