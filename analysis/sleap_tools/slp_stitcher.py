# Script for stitching together .slp files.
# Written by S. Wolf
import glob
import re
from time import time
import sleap
import h5py
import rich
import argparse


class RateColumn(rich.progress.ProgressColumn):
    """Renders the progress rate."""

    def render(self, task: "Task") -> rich.progress.Text:
        """Show progress rate."""
        speed = task.speed
        if speed is None:
            return rich.progress.Text("?", style="progress.data.speed")
        return rich.progress.Text(f"{speed:.2f} it/s", style="progress.data.speed")

parser = argparse.ArgumentParser(
        description="Import all .slp files in a folder, stitch together the contents."
    )

parser.add_argument(
        "folder_path",
        help="Path of folder containing all of the .slp files to be stitched togther.",
        type=str,
    )

args = parser.parse_args()

filenames = glob.glob(
    args.folder_path + "*.slp"
)
filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
print(filenames)

base_labels = sleap.Labels.load_file(filenames[0])

report_period = 0.1
with rich.progress.Progress(
    "{task.description}",
    rich.progress.BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "ETA:",
    rich.progress.TimeRemainingColumn(),
    RateColumn(),
    auto_refresh=False,
    refresh_per_second=5,
    speed_estimate_period=10,
) as progress:
    task = progress.add_task("Running...", total=len(filenames[1::]))
    last_report = time()
    for filename in filenames[1::]:
        progress.update(task, advance=1)
        elapsed_since_last_report = time() - last_report
        if elapsed_since_last_report > report_period:
            progress.refresh()
        new_labels = sleap.Labels.load_file(filename)
        base_labels.labeled_frames.extend(new_labels.labeled_frames)
        base_labels.merge_matching_frames()
        base_labels._update_from_labels(merge=True)


# base_labels = sleap.Labels.complex_merge_between(base_labels, new_labels.labeled_frames)
sleap.Labels.save_file(base_labels, "merged.slp")
print(f"Saved merged file as ./merged.slp")

f = h5py.File("merged.slp", "r+")
print(len(f["frames"]["video"][:]))
print(f["frames"]["video"][:])
f["frames"]["video"] = 0
print(len(f["frames"]["video"][:]))
print(f["frames"]["video"][:])
f.close()