import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description="Import two CSV files as dataframes, and print out difference locations + stats.  Intended for csv files from /aruco_tools/matching_pipeline.py, with suffix _matching_result.csv"
)

parser.add_argument(
    "CSV_path",
    help="Filepath of the matching result csv file spit out by matching_pipeline.py",
    type=str,
)

parser.add_argument(
    "tag_number",
    help="Tag number of bee we want to calculate the autocorrelation for.",
    type=int,
)

parser.add_argument(
    "frame_number",
    help="Frame number from which to calculate autocorrelation",
    type=int,
)

args = parser.parse_args()

# Read in CSV file as dataframe
coords_df = pd.read_csv(args.CSV_path).set_index("Frame")
# Isolate the data for the specific tag number
coords_df = coords_df[coords_df["Tag"] == args.tag_number]
# Fill missing indices with NaN
frame_numbers = coords_df.index
new_index = np.arange(np.min(frame_numbers), np.max(frame_numbers) + 1)
coords_df = coords_df.reindex(new_index)
# Linear interpolation for missing values
coords_df.interpolate(
    method="linear",
    limit_area="inside",
    axis=0,
    limit=20,
    inplace=True,
    downcast="infer",
)
print(coords_df)

theta = np.arctan2(
    coords_df["headY"][args.frame_number] - coords_df["thoraxY"][args.frame_number],
    coords_df["headX"][args.frame_number] - coords_df["thoraxX"][args.frame_number],
)
print(f"\nCurrent frame theta: {theta}")

angular_resolution = 100
thetas_to_check = np.linspace(
    theta, theta + 2 * np.pi(), num=angular_resolution, endpoint=False
)
cross_cor = np.zeros((max_shift, angular_resolution))

angular_index = -1
for current_theta in thetas_to_check:
    angular_index += 1
    # Convert coordinates
    cX_array = coords_df["cX"].to_numpy()
    cY_array = coords_df["cY"].to_numpy()

    cX_array = cX_array * np.cos(theta)  # Forward-facing
    cY_array = cY_array * np.cos(theta)  # Side-facing

    max_shift = 1000

    for shift in range(0, max_shift):
        cross_cor[shift, angular_index] = np.correlate(
            cX_array[args.frame_number : -1 - max_shift], cX_array
        )
