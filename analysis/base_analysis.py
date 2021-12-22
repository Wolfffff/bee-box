# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import logging
from seaborn.distributions import distplot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import palettable
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import pandas as pd
import joypy
import h5py
import numpy as np
from pathlib import Path
import os

wd = '/Genomics/ayroleslab2/scott/bees/bee-box/analysis'
os.chdir(wd)


# data_dir = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data"

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis_logger")

import utils.trx_utils as trx_utils
bee_tracks_df = pd.read_csv("../data/tracking_results/20210909_run000_00000000_48Hrs_libx264/20210909_run000_00000000_48Hrs_libx264_000/20210909_run000_00000000_48Hrs_libx264_000_aruco_data_with_track_numbers.csv")

# %%
bee_tracks_df = bee_tracks_df[["Frame", "Tag", "abdomenX","abdomenY"]]
bee_tracks_df_x = bee_tracks_df[bee_tracks_df.Tag != -1].pivot(index='Frame', columns='Tag', values='abdomenX')
bee_tracks_df_y = bee_tracks_df[bee_tracks_df.Tag != -1].pivot(index='Frame', columns='Tag', values='abdomenY')
# %%
bee_tracks = np.stack((bee_tracks_df_x.to_numpy(), bee_tracks_df_y.to_numpy()),axis=1)
bee_tracks[:,1,:] = -bee_tracks[:,1,:]
# %%
bee_tracks
missing_ct = np.count_nonzero(np.isnan(bee_tracks[:,0,:]),axis=0)
missing_freq = missing_ct/bee_tracks.shape[0]
bee_tracks = bee_tracks[:,:,missing_freq < 0.5]
# %%
frame_count, node_count,instance_count = bee_tracks.shape
bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear")
bee_vel = trx_utils.instance_node_velocities_bees(bee_tracks,0,bee_tracks.shape[0])
mask = np.stack(((bee_vel > 100),(bee_vel > 100)), axis=1)
bee_tracks[mask] = np.nan

bee_tracks = trx_utils.fill_missing(bee_tracks, kind="linear")
bee_tracks = trx_utils.smooth_median(bee_tracks,window=5)
# bee_tracks = trx_utils.smooth_ewma(bee_tracks,alpha=0.3)


start_frame = 0
end_frame = int(20*60)
for bee_id in range(instance_count):
    plt.plot(bee_tracks[start_frame:end_frame,0,bee_id],bee_tracks[start_frame:end_frame,1,bee_id],alpha = 0.1)
# %%
import distinctipy
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from cv2 import cv2
from matplotlib.animation import FuncAnimation
import palettable

cap = cv2.VideoCapture("/Genomics/ayroleslab2/scott/bees/bee-box/data/tracking_results/20210909_run000_00000000_48Hrs_libx264/20210909_run000_00000000_48Hrs_libx264_000/20210909_run000_00000000_48Hrs_libx264_000_annotated.mp4")


colors = distinctipy.get_colors(instance_count)
# start_frame= 0
# end_frame=1000
# frame_numbers = list(range(start_frame, end_frame))
def animate(i):
    plt.cla()
    frame_idx =  i + 5000

    print(f"Saving frame {i}")
    for bee_idx in range(instance_count):
        ax.axis("off")
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.patch.set_visible(False)
        data = bee_tracks[:,:,bee_idx]

        trail_length = 20
        data_subset = data[(frame_idx - trail_length) : frame_idx,:]
        for idx in range(
            data_subset.shape[0]
        ):  # Note that you need to use single steps or the data has "steps"
            plt.plot(
                data_subset[idx : (idx + 2), 0]/2,
                -data_subset[idx : (idx + 2), 1]/2,
                linewidth=1*idx / trail_length,
                color=colors[bee_idx],
                alpha=1
            )
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx + 5)
    res, frame = cap.read()
    frame = frame[:, :, 0]
    plt.imshow(frame, cmap="gray", zorder=0)



fig, ax = plt.subplots()
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
# fig.set_size_inches(5, 5, True)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax.axis("off")
# fig.patch.set_visible(False)
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

plt.rcParams['animation.ffmpeg_path'] = '/Genomics/argo/users/swwolf/.conda/envs/sleap_dev/bin/ffmpeg'
ani = FuncAnimation(fig, animate, frames=range(100,600)).save(
    "withframe.mp4",
    fps=20,
    bitrate=20000,
    dpi=300,
    writer="ffmpeg",
    codec="libx264",
    extra_args=["-preset", "veryslow", "-pix_fmt", "yuv420p"]
)
# %%
