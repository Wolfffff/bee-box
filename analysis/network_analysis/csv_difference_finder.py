import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
        description="Import two CSV files as dataframes, and print out difference locations + stats.  Intended for csv files from /aruco_tools/matching_pipeline.py, with suffix _matching_result.csv"
    )

parser.add_argument(
        "CSV1_path",
        help="Filepath of first csv file (order doesn't matter).",
        type=str,
    )

parser.add_argument(
        "CSV2_path",
        help="Filepath of second csv file (order doesn't matter).",
        type=str,
    )

args = parser.parse_args()

df1 = pd.read_csv(args.CSV1_path)
df2 = pd.read_csv(args.CSV2_path)

df1.sort_index(inplace=True)
df2.sort_index(inplace=True)

frames1 = np.sort(np.unique(df1['Frame'].to_numpy()))
frames2 = np.sort(np.unique(df2['Frame'].to_numpy()))

print(f'CSV 1 frame range: {np.min(frames1), np.max(frames1)}')
print(f'CSV 2 frame range: {np.min(frames1), np.max(frames1)}\n')

print(f'CSV 1 missing frames: {np.setdiff1d(np.arange(np.min(frames1), np.max(frames1)), frames1)}')
print(f'CSV 1 missing frames: {np.setdiff1d(np.arange(np.min(frames2), np.max(frames2)), frames2)}\n')

for frame in np.intersect1d(frames1, frames2):
	# print('=' * 80)
	# print(f'Frame: {frame}\n')
	frames_from_df1 = df1.loc[df1['Frame']==frame]
	frames_from_df2 = df2.loc[df2['Frame']==frame]
	
	differences = 0
	for tag in frames_from_df1['Tag'].to_numpy():
		if str(frames_from_df1.loc[frames_from_df1['Tag'] == tag]) != str(frames_from_df2.loc[frames_from_df2['Tag'] == tag]):
			differences += 1
			# print(f'Difference detected: tag {tag}')
			# try:
			# 	print(frames_from_df1.loc[frames_from_df1['Tag'] == tag])
			# 	print(frames_from_df2.loc[frames_from_df1['Tag'] == tag])
			# except:
			# 	pass
			# print('\n')

	# print('\n')

total_lines = len(df1['Frame'].to_numpy())
print(f'Total of {differences} differences out of {total_lines} total')