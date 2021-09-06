# Stats on inter-bee distance

import time
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
			print(f'[Long interaction] Bee {tag_to_check} interacted with bee {interaction[1]} for {interaction[0]} frames ending in frame {interaction[2]}')

	plt.hist(output_data, bins = 100)
	plt.yscale('log')
	plt.show()

def print_all_long_interactions(csv_coordinates_path: str, output_path: str = './long_interactions_list.csv', maximum_distance_for_interaction_threshold: float = 400., interaction_print_threshold: int = 100, frame_rate: float = 20, fill_nans: int = 10):
	'''
	Go through coordinate CSV files from ../aruco_tools/matching_pipeline.py (should end with '_aruco_data_with_track_numbers.csv') and save all of the long interactions.
	Results will be outputted as a dataframe in the form of a csv file.
	First column: tuple with the two bee tag numbers for the interacting bees.  Smaller tag number comes first.
	Second column: interaction start timestamp
	Third column: interaction end timestamp

	Args:
		csv_coordinates_path: the path to the csv file from ../aruco_tools/matching_pipeline.py (should end with '_aruco_data_with_track_numbers.csv').
		output_path: output path of csv file containing the interaction data
		maximum_distance_for_interaction_threshold: maximum distance between bees to be counted as a frame within an interaction
		interaction_print_threshold: Minimum duration of interaction in frames to be outputted.
		frame_rate: framerate of the video
		fill_nans: largest number of consecutive NaNs to fill.
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
		f"\nRemoved redundant column(s), set frames as index, filled up to {fill_nans} consecutive NaN sports: \n"
		+"---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n\n"
	)
	columns_to_fill = ['cX', 'cY']
	coords_df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].interpolate(method = 'linear', limit = fill_nans)

	coords_df = coords_df.loc[coords_df['Tag'] > 0]
	print(
		"\nRemoved unidentified bee data points: \n"
		+ "---------------------------------------------- \n"
		+ str(coords_df.head())
		+ "\n\n"
	)

	# Collect distances
	distances = []
	interactions_list = []
	frames = coords_df.index.values
	unique_frames = np.sort(np.unique(frames))
	print('Computing distances...')

	unique_tags = np.sort(pd.unique(coords_df['Tag']))
	print(unique_tags)

	for tag_to_check in unique_tags:
		# Dict of interaction lengths by tag number.  Key is tag number, value is interaction lengths.
		current_interaction_partners = {}

		# List of interactions that finished: [interaction length in frames, interaction partner bee tag number, end of interaction frame number]
		finished_interactions = []

		print(f'Checking interactions between tag {tag_to_check} and larger tag numbers.')
		for frame in progressBar(unique_frames):
			# Uncomment for debugging to speed things up
			# if frame > 1000:
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
					if tag_number > tag_to_check:
						if (tag_number not in current_interaction_partners.keys()):
							# If there's a tag showing up in the data that isn't in our dict of interaction lengths, add it.
							current_interaction_partners[tag_number] = 0
						if distance <= maximum_distance_for_interaction_threshold:
							current_interaction_partners[tag_number] += 1
						elif current_interaction_partners[tag_number] > 0:
								finished_interactions.append([current_interaction_partners[tag_number], tag_number, frame])
								current_interaction_partners[tag_number] = 0

		for interaction in finished_interactions:
			# Print longest distributions:
			if interaction[0] > interaction_print_threshold:
				if interaction[2] - interaction[1] == 0:
					start_time = '00:00:00'
				else:
					start_time = time.strftime('%H:%M:%S', time.gmtime((interaction[2] - interaction[1]) / frame_rate))
				end_time = time.strftime('%H:%M:%S', time.gmtime((interaction[2]) / frame_rate))
				print(f'[Long interaction] Bee {tag_to_check} interacted with bee {interaction[1]} during the interval {start_time} - {end_time}')
				interactions_list.append(((int(tag_to_check), int(interaction[1])), str(start_time), str(end_time)))

	interactions_df = pd.DataFrame(interactions_list, columns = ["Bees", "Start", "END"])
	print('\n')
	print(interactions_df)

	# Save
	interactions_df.to_csv(output_path)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(
	    description="Utilities to help understand raw output from matching pipeline.  Adjust parameters in code.  Command line is just something to make it easy to call the code on multiple files."
	)
	parser.add_argument(
	    "csv_path",
	    help="the path to the csv file from ../aruco_tools/matching_pipeline.py (should end with '_aruco_data_with_track_numbers.csv').",
	    type=str,
	)
	parser.add_argument(
		"processing_step",
		help="how to process the raw data from the input csv.",
		choices=["distance_distribution", "interaction_length_distribution", "print_all_long_interactions"],
	)

	args = parser.parse_args()

	if args.processing_step == 'distance_distribution':
		distance_distribution(args.csv_path, 31)
	elif args.processing_step == 'interaction_length_distribution':
		interaction_length_distribution(args.csv_path, 31, maximum_distance_for_interaction_threshold = 400.)
	elif args.processing_step == 'print_all_long_interactions':
		print_all_long_interactions(args.csv_path)