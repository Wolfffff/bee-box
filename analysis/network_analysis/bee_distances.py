# Stats on inter-bee distance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(coords_1: tuple, coords2: tuple):
	# Pass in two (x, y) and to recieve the euclidean distance between them
	return np.sqrt(np.power(coords_1[0] - coords2[0], 2) + np.power(coords_1[1] - coords2[1], 2))

def progressBar(
	iterable, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"
):
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
	def printProgressBar(iteration):
		percent = ("{0:." + str(decimals) + "f}").format(
			100 * (iteration / float(total))
		)
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + "-" * (length - filledLength)
		print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)

	# Initial Call
	printProgressBar(0)
	# Update Progress Bar
	for i, item in enumerate(iterable):
		yield item
		printProgressBar(i + 1)
	# Print New Line on Complete
	print()

def distance_distribution(csv_coordinates_path: str, tag_to_check: int, notification_threshold: float = 30.):
	'''
	Look at the distribution of distances from one bee to all of the rest of the bees.
	Display result as histogram.

	Args:
		csv_coordinates_path: Path to the csv file output from ../aruco_tools/matching_pipeline.py.  Should end with '_aruco_data_with_track_numbers.csv'.
		tag_to_check: The bee for which we want to see the distribution of distances to all other bees.
		notification_threshold: Bees that are too close together might be indicative of bad data.  Distance data points that are below the value set in this variable trigger a message to the console to aid diagnosis.
	'''
	# Load data
	coords_df = pd.read_csv(csv_coordinates_path)
	print(
		"\nLoaded csv file with following head: \n"
		+ "---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n"
	)

	coords_df = coords_df.reindex(columns=["Frame", "Tag", "thoraxX", "thoraxY"])
	coords_df = coords_df.set_index("Frame")
	print(
		"\nRemoved redundant column(s), set frames as index: \n"
		+ "---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n\n"
	)

	coords_df = coords_df.loc[coords_df['Tag'] > 0]
	print(
		"\nRemoved unidentified bee data points: \n"
		+ "---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n\n"
	)

	# Collect distances
	distances = []
	frames = coords_df.index.values
	unique_frames = np.sort(np.unique(frames))
	print('Computing distances...')
	for frame in progressBar(unique_frames):
		# print('\n')
		frame_df = coords_df.loc[frame]
		tag_to_check_df = frame_df.loc[frame_df['Tag'] == tag_to_check]
		# If we have data for the tag to check
		# print(str(tag_to_check_df))
		if len(tag_to_check_df.index) == 1:
			tag_to_check_coords = (tag_to_check_df.loc[frame]['thoraxX'], tag_to_check_df.loc[frame]['thoraxY'])
			frame_df = frame_df.loc[frame_df['Tag'] != tag_to_check]
			# print(str(frame_df))
			# Iterate through the remaining rows and collect distances
			for row in frame_df.itertuples():
				distance = euclidean_distance((row.thoraxX, row.thoraxY), tag_to_check_coords)
				distances.append(distance)
				if distance < notification_threshold:
					try:
						tag_number = int(row.Tag)
						print(f'[POSSIBLE SPURIOUS DATA POINT] The distance between bee {int(tag_to_check)} and bee {tag_number} on frame {frame} is {np.round(distance, 2)} pixels.          ')
					except Exception as e:
						print(f'\n{e}')
						tag_object = row.Tag
						print(f'\n{tag_object}')

	print(f"Total data points: {len(distances)}")

	idx = 0
	distances_to_look_out = []
	# for idx in range(len(distances)):
	# 	if distances[idx] < 100:
	# 		distances_to_look_out.append(distances[idx])

	plt.hist(np.log10(distances), bins = 100)
	# plt.hist(distances_to_look_out, bins = 100)
	plt.show()

def interaction_length_distribution(csv_coordinates_path: str, tag_to_check: int, maximum_distance_for_interaction_threshold: float = 400., minimum_interaction_length_frames: int = 10, interaction_print_threshold: int = 500):
	# Load data
	coords_df = pd.read_csv(csv_coordinates_path)
	print(
		"\nLoaded csv file with following head: \n"
		+ "---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n"
	)

	coords_df = coords_df.reindex(columns=["Frame", "Tag", "thoraxX", "thoraxY"])
	coords_df = coords_df.set_index("Frame")
	print(
		"\nRemoved redundant column(s), set frames as index: \n"
		+ "---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n\n"
	)

	coords_df = coords_df.loc[coords_df['Tag'] > 0]
	print(
		"\nRemoved unidentified bee data points: \n"
		+ "---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n\n"
	)

	# Collect distances
	distances = []
	frames = coords_df.index.values
	unique_frames = np.sort(np.unique(frames))
	print('Computing distances...')

	# Dict of interaction lengths by tag number.  Key is tag number, value is interaction lengths.
	current_interaction_partners = {}

	# List of interactions that finished: [interaction length in frames, interaction partner bee tag number, end of interaction frame number]
	finished_interactions = []

	for frame in progressBar(unique_frames):
		# Uncomment for debugging to speed things up
		# if frame > 10000:
		# 	break
		# print('\n')
		frame_df = coords_df.loc[frame]
		tag_to_check_df = frame_df.loc[frame_df['Tag'] == tag_to_check]
		# If we have data for the tag to check
		# print(str(tag_to_check_df))
		if len(tag_to_check_df.index) == 1:
			tag_to_check_coords = (tag_to_check_df.loc[frame]['thoraxX'], tag_to_check_df.loc[frame]['thoraxY'])
			frame_df = frame_df.loc[frame_df['Tag'] != tag_to_check]
			# print(str(frame_df))
			# Iterate through the remaining rows and collect distances
			for row in frame_df.itertuples():
				distance = euclidean_distance((row.thoraxX, row.thoraxY), tag_to_check_coords)
				tag_number = int(row.Tag)
				if tag_number not in current_interaction_partners.keys():
					# If there's a tag showing up in the data that isn't in our dict of interaction lengths, add it.
					current_interaction_partners[tag_number] = 0
				if distance <= maximum_distance_for_interaction_threshold:
					current_interaction_partners[tag_number] += 1
				elif current_interaction_partners[tag_number] > minimum_interaction_length_frames:
					finished_interactions.append([current_interaction_partners[tag_number], tag_number, frame])
					current_interaction_partners[tag_number] = 0

	print(finished_interactions)
	output_data = []
	for interaction in finished_interactions:
		output_data.append(interaction[0])

		# Print longest distributions:
		if interaction[0] > interaction_print_threshold:
			print(f'[Long interaction] Bee {tag_to_check} interacted with bee f{interaction[1]} for {interaction[0]} frames ending in frame {interaction[2]}')

	plt.hist(output_data, bins = 100)
	plt.yscale('log')
	plt.show()

if __name__ == '__main__':
	# distance_distribution('d:/crop_matching_aruco_data_with_track_numbers.csv', 31)
	interaction_length_distribution('d:/crop_matching_aruco_data_with_track_numbers.csv', 31, maximum_distance_for_interaction_threshold = 400.)