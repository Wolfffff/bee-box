# Utilities for working with ArUco tag data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math
import pickle  #pip install pickle-mixin
from tabulate import tabulate


def load_into_pd_dataframe(ArUco_csv_path: str) -> pd.DataFrame:
	"""
		Read in CSV and index by frame number
	"""
	aruco_df = pd.read_csv(ArUco_csv_path)
	print('\nLoaded csv file with following head: \n' + \
			'---------------------------------------------- \n' + str(aruco_df.head()) + '\n')

	aruco_df = aruco_df[['Frame', 'Tag', 'cX', 'cY', 'Theta']]
	aruco_df = aruco_df.set_index('Frame')
	print('\nRemoved redundant column(s), set frames as index: \n' + \
			'---------------------------------------------- \n' + str(aruco_df.head()) + '\n\n')

	return aruco_df.sort_index()

def find_tags_fast(aruco_df: pd.DataFrame) -> np.array:
	"""
	Get unique tags from data
	"""
	tags = pd.unique(aruco_df['Tag'])
	return tags

def sort_for_individual_tags(aruco_df: pd.DataFrame, tags: np.array) -> dict:
	"""
	Dict of data frames organized by tag

	It may be more efficient to use a 3 dim array here but this allows a few niceties and doesnt require a large array with many NAs
	"""
	aruco_dict_by_tag = {}
	for tag in tags:
		aruco_dict_by_tag[tag] = aruco_df.loc[aruco_df['Tag'] == tag]
		
		# Remove tag column as it would be redundant 
		aruco_dict_by_tag[tag].drop('Tag', 1)

	return aruco_dict_by_tag
   
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
	# This code is taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
	# in the answer written by user Greenstick.
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
		printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
	"""
	total = len(iterable)
	# Progress Bar Printing Function
	def printProgressBar (iteration):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
	# Initial Call
	printProgressBar(0)
	# Update Progress Bar
	for i, item in enumerate(iterable):
		yield item
		printProgressBar(i + 1)
	# Print New Line on Complete
	print()

def interpolate_empties(aruco_dict_by_tag: dict, tags: np.array) -> dict:
	"""
	Use linear interpolation to fill in missing data 
	Note that there's no maximum gap length so we will fill any missing data
	"""
	for tag in tags:
		frames_with_data = np.asarray(aruco_dict_by_tag[tag].index.values)
		previous_frame = frames_with_data[0]

		# Interpolated coordinates 
		interp_X = []
		interp_Y = []
		indices = []

		# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html 	
		for current_frame in progressBar(frames_with_data, prefix = 'Tag #' + str(tag) + ' interpolation', length = 50):
			if current_frame - previous_frame > 1:
				n_to_interp = current_frame - previous_frame - 1
				start_point_x = aruco_dict_by_tag[tag]['cX'][previous_frame]
				start_point_y = aruco_dict_by_tag[tag]['cY'][previous_frame]
				end_point_x = aruco_dict_by_tag[tag]['cX'][current_frame]
				end_point_y = aruco_dict_by_tag[tag]['cY'][current_frame]
				


				x_inc = (start_point_x - end_point_x) / float(n_to_interp)
				y_inc = (start_point_y - end_point_y) / float(n_to_interp)
				for interp_frame in range(previous_frame + 1, current_frame):
					interp_X.append(interp_X[-1] + x_inc)
					interp_Y.append(interp_Y[-1] + y_inc)
					indices.append(interp_frame)
			
			interp_X.append(aruco_dict_by_tag[tag]['cX'][current_frame])
			interp_Y.append(aruco_dict_by_tag[tag]['cY'][current_frame])
			indices.append(current_frame)

			previous_frame = current_frame

		interp_data = np.transpose(np.array([interp_X, interp_Y]))
		aruco_dict_by_tag[tag] = pd.DataFrame(data = interp_data, columns = ['cX', 'cY'], index = indices)
		aruco_dict_by_tag[tag].index.name = 'Frame'
		print('\n' + str(aruco_dict_by_tag[tag].head()) + '\n\n')

	return aruco_dict_by_tag

def save_dict(dict_of_dfs_to_save: dict, path:str) -> None:
	"""
	Dump dict of dataframes
	"""
	pickle.dump(dict_of_dfs_to_save, open(path, 'wb'))

def reload_saved_dataframes(path: str) -> dict:
	"""
	Read dict of dataframes
	"""
	return pickle.load(open(path, 'rb'))

def threshold_remove_bad_tags(aruco_df: pd.DataFrame, total_frames: int, percent_threshold: float) -> pd.DataFrame:
	tags = find_tags_fast(aruco_df)
	aruco_dict_by_tag = sort_for_individual_tags(aruco_df, tags)
	_, _, _, _, _, captured_percents = collect_all_bee_stats(tags, aruco_dict_by_tag, total_frames, 1)

	print(f'\n\n--------------------------------------------------\n\nRemoving tags under the {percent_threshold}% detection rate threshold:\n')
	for idx in range(len(tags)):
		if captured_percents[idx] < percent_threshold:
			aruco_df = aruco_df[aruco_df.Tag != tags[idx]]
			print(f'Removed tag {tags[idx]}')
	print('\n\n')

	return aruco_df


def find_tag_path_with_gaps(aruco_dict_by_tag:dict, tag_number:int):
	"""
	Take paths with gaps and annotate with gap length
	"""
	
	path = [[],[],[],[]]
	
	path[2] = aruco_dict_by_tag[tag_number].index
	path[0] = aruco_dict_by_tag[tag_number]['cX']
	path[1] = aruco_dict_by_tag[tag_number]['cY']

	path[3] = np.zeros(len(path[2]))

	path = np.array(path).astype('float')
	path[3, 0] = 1
	for j in range(1, len(path[2])):
		path[3, j] = path[2, j] - path[2, j - 1]

	return path
			
def collect_all_bee_stats(tags, aruco_df_by_tag, total_frames, fps):
	"""
	
	"""
	longest_skips = []
	g1s_skips = []
	g10s_skips = []
	g100s_skips = []
	captured_percents = []

	for tag_number in tags:
		path_wg = find_tag_path_with_gaps(aruco_df_by_tag, tag_number)
		skips_data = np.array(path_wg[3]).astype('int')

		longest_skips.append(np.max(skips_data))

		g1s_skips.append(len([skip for skip in skips_data if skip > fps]))
		g10s_skips.append(len([skip for skip in skips_data if skip > fps * 10]))
		g100s_skips.append(len([skip for skip in skips_data if skip > fps * 100]))

		captured_percents.append(float(len(aruco_df_by_tag[tag_number].index)) * 100. / total_frames)

	return tags, longest_skips, g1s_skips, g10s_skips, g100s_skips, captured_percents

def display_bee_stats_table(aruco_df: pd.DataFrame, aruco_dict_by_tag: dict, total_frames: int, fps: float):
	"""
	
	"""
	tags, longest_skips, g1s_skips, g10s_skips, g100s_skips, captured_percents = collect_all_bee_stats(aruco_df, aruco_dict_by_tag, total_frames, fps)

	longest_skips = [str(j) for j in longest_skips]
	longest_skips.insert(0, ' Longest gaps')

	g1s_skips = [str(j) for j in g1s_skips]
	g1s_skips.insert(0, ' >   1 second gaps')

	g10s_skips = [str(j) for j in g10s_skips]
	g10s_skips.insert(0, ' >  10 second gaps')

	g100s_skips = [str(j) for j in g100s_skips]
	g100s_skips.insert(0, ' > 100 second gaps')

	captured_percents = [str(round(j, 2)) for j in captured_percents]
	captured_percents.insert(0, '% frames with data')

	table = [captured_percents, longest_skips, g1s_skips, g10s_skips, g100s_skips]

	print(tabulate(table, headers = tags))

def print_by_tag_data(tags, aruco_df_by_tag):
	"""
	
	"""
	for tag in tags:
		print('\n' + '=' * 50)
		print(tag)
		print('\n' + '-' * 50)
		print(aruco_df_by_tag[tag])

def save_df(aruco_df: pd.DataFrame, filepath: str) -> None:
	aruco_df.to_csv(filepath)



if __name__ == "__main__":
	aruco_df = load_into_pd_dataframe('d:\\20210706_run000_00000000_aruco_annotated.csv')
	tags = np.sort(find_tags_fast(aruco_df))
	aruco_df_by_tag = sort_for_individual_tags(aruco_df, tags)
	print_by_tag_data(tags, aruco_df_by_tag)
	print('--------------------------------------------------\n\nStats by tags: \n')
	display_bee_stats_table(tags, aruco_df_by_tag, np.max(aruco_df.index), 20)

	aruco_df = threshold_remove_bad_tags(aruco_df, np.max(aruco_df.index), 30.)

	tags = np.sort(find_tags_fast(aruco_df))
	aruco_df_by_tag = sort_for_individual_tags(aruco_df, tags)
	# print_by_tag_data(tags, aruco_df_by_tag)
	display_bee_stats_table(tags, aruco_df_by_tag, np.max(aruco_df.index), 20)

	save_df(aruco_df, 'd:\\20210706_run000_00000000_aruco_annotated_30p_threshold.csv')