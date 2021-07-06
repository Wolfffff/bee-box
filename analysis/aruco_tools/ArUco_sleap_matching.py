import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tabulate import tabulate
from munkres import Munkres

import aruco_utils_pd as awpd

# define names of a few possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}

def ArUco_cleaner(video_path: str, frame_array: list, tl_coords: tuple[int, int], square_dim: int, bee_population: int) -> 'DataFrame':
	"""Processes a video for ArUco tags, attempting to iteratively lower the error correction to flush out false positives and misreadings.	"""

	# load the ArUCo dictionary and grab the ArUCo parameters
	print("[INFO] detecting '{}' tags...".format("DICT_5X5_100"))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
	arucoParams = cv2.aruco.DetectorParameters_create()
	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	# arucoParams.adaptiveThreshWinSizeStep = 1
	# arucoParams.adaptiveThreshWinSizeMin = 3
	# arucoParams.adaptiveThreshWinSizeMax = 30
	# arucoParams.adaptiveThreshConstant = 12
	# arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	vs = cv2.VideoCapture(video_path)
	fps = vs.get(cv2.CAP_PROP_FPS)
	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])

	detected = 0

	# Start by casting a wide net
	arucoParams.errorCorrectionRate = 1.

	for frame_idx in awpd.progressBar(frame_array):
		# Navigate to relevant frame in video
		vs.set(1, frame_idx)
		ret, frame = vs.read()

		# In-built cropping feature; cropping video is slow so best to just crop the frames we need, as needed
		frame = frame[tl_coords[0] : tl_coords[0] + square_dim, tl_coords[1] : tl_coords[1] + square_dim]


		# Detect ArUco markers in the input frame
		(corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters = arucoParams)

		# Sort out data into dataframe for easy access
		detected += len(corners)
		if len(corners) > 0:
			for (markerCorner, markerID) in zip(corners, ids):
				corners = markerCorner.reshape((4, 2))
				(topLeft, topRight, bottomRight, bottomLeft) = corners

				# convert each of the (x, y)-coordinate pairs to integers
				topRight = (topRight[0], topRight[1])
				bottomRight = (bottomRight[0], bottomRight[1])
				bottomLeft = (bottomLeft[0], bottomLeft[1])
				topLeft = (topLeft[0], topLeft[1])
				
				# Calculate centroid x a
				cX = (topLeft[0] + bottomRight[0]) / 2.
				cY = (topLeft[1] + bottomRight[1]) / 2.

				# 'Frame', 'Tag', 'cX','cY'
				results_df.loc[len(results_df)] = [int(frame_idx), int(markerID[0]), cX, cY]

	# Sort out which tags have been detected; sort by amount of data
	tags = awpd.find_tags_fast(results_df)
	aruco_df_by_tag = awpd.sort_for_individual_tags(results_df, tags)

	tag_lengths = {}
	for tag in tags:
		tag_lengths[int(tag)] = len(aruco_df_by_tag[tag].index)

	print(tag_lengths)
	print('Total of ' + str(len(tags)) + ' tags detected in first pass')

	pre_suspicion_data = len(results_df.index)

	
	suspicious_index = 0
	sorted_tag_lengths = sorted(tag_lengths.items(), key = lambda item: item[1])
	if len(tags) > bee_population:
		while suspicious_index < len(tags):
			print('\n\nToo many tags... ' + str(len(tags)) + '/' + str(bee_population))
			suspicious_tag = sorted_tag_lengths[suspicious_index][0]
			print("I'm suspicious of tag " + str(suspicious_tag))
			suspicious_frames = aruco_df_by_tag[suspicious_tag]['Frame'].tolist()
			print("Suspicious frames: " + str(suspicious_frames))

			# Start by getting rid of data from suspicious frames.
			for j in suspicious_frames:
				results_df = results_df[results_df.Frame != j]
			aruco_df_by_tag = awpd.sort_for_individual_tags(results_df, tags)
			suspicious_index += 1

			for frame_idx in suspicious_frames:
				arucoParams.errorCorrectionRate = 1.
				while arucoParams.errorCorrectionRate > 0.1:
					vs.set(1, frame_idx)
					ret, frame = vs.read()
					frame = frame[tl_coords[0] : tl_coords[0] + square_dim, tl_coords[1] : tl_coords[1] + square_dim]
					(corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters = arucoParams)

					if suspicious_tag not in ids:
						break
					else:
						arucoParams.errorCorrectionRate -= 0.2
				print('Refinished frame ' + str(frame_idx) + ' with errorCorrectionRate = ' + str(round(arucoParams.errorCorrectionRate, 1)))
				print('Tags identified after refinishing: ' + str(np.sort(ids.flatten())))

				if len(corners) > 0:
					for (markerCorner, markerID) in zip(corners, ids):
						corners = markerCorner.reshape((4, 2))
						(topLeft, topRight, bottomRight, bottomLeft) = corners

						# convert each of the (x, y)-coordinate pairs to integers
						topRight = (topRight[0], topRight[1])
						bottomRight = (bottomRight[0], bottomRight[1])
						bottomLeft = (bottomLeft[0], bottomLeft[1])
						topLeft = (topLeft[0], topLeft[1])
						
						# Calculate centroid x a
						cX = (topLeft[0] + bottomRight[0]) / 2.
						cY = (topLeft[1] + bottomRight[1]) / 2.

						# 'Frame', 'Tag', 'cX','cY'
						results_df.loc[len(results_df)] = [frame_idx, int(markerID[0]), cX, cY]

			tags = awpd.find_tags_fast(results_df)
			print(tags)
			if len(tags) <= bee_population:
				print('Bee population now matched... terminating!')
				print(results_df)
				break
	post_suspicion_data = len(results_df.index)
	print('Pre -suspicion data amount:', pre_suspicion_data)
	print('Post-suspicion data amount:', post_suspicion_data)
	print('Lost data amount: ', post_suspicion_data - pre_suspicion_data)

	return results_df

def sleap_reader(slp_predictions_path: str) -> h5py.File:
	f = h5py.File(slp_predictions_path, 'r')
	keys = list(f.keys())
	print(keys)

	for key in keys:
		sleap_data = np.array(f[key])
		print(key)
		print(sleap_data.shape)
		if len(sleap_data) >= 3:
			print(sleap_data[0:3])
			print(np.asarray(sleap_data[0]).dtype)
		print('\n')

	return f


def proximity_matching(cleaned_aruco_csv_path: str, slp_predictions_path: str, maximum_allowable_matching_distance: int) -> list:
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	tag_detection_count = [np.sum(aruco_df.Tag == tag) for tag in tags]

	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])


	unique_tracks = np.sort(np.unique([j[4] for j in sleap_instances]))
	# (number of tracks) x (number of aruco tags)
	track_identity_votes = np.zeros((len(unique_tracks), len(tags)))

	print('\n')
	print('Unique tracks:              ', unique_tracks)
	print('Unique tags:                ', tags)

	last_frame_aruco = np.max(aruco_df.index)
	last_frame_sleap = sleap_instances[-1][2]
	last_frame = int(np.min([last_frame_aruco, last_frame_sleap]))
	print('Last frame with ArUco data: ', last_frame_aruco)
	print('Last frame with SLEAP data: ', last_frame_sleap)
	print('Overall last frame:         ', last_frame)
	print('\n')

	instances_idx = 0
	for row in aruco_df.itertuples():
		current_frame = row.Index

		# Break if we've reached the end
		if current_frame == last_frame:
			break

		# Grab as a tuple the nth instance entry in the .slp predictions file
		nth_inst_tuple = sleap_instances[instances_idx]
		
		# Find starting point in .slp instances data
		while nth_inst_tuple[2] != current_frame:  # frame_id
			instances_idx += 1
			nth_inst_tuple = sleap_instances[instances_idx]

		# Put instances for current frame in a list
		# Technically this does some redundant computation, but I'll leave it in for interest of simplicity
		# TODO: optimize this code
		instance_frame = current_frame
		temp_idx = instances_idx
		current_frame_instances = []
		while instance_frame == current_frame:
			current_frame_instances.append(nth_inst_tuple)
			temp_idx += 1
			nth_inst_tuple = sleap_instances[temp_idx]
			instance_frame = nth_inst_tuple[2]  # frame_id

		# Find nearest track to current tag and vote
		distances = []
		for instance in current_frame_instances:
			# For simplicity, find mean position of points in instance
			start_idx = instance[7]
			end_idx = instance[8]
			pX = 0.
			pY = 0.
			points = 0.
			for l in range(start_idx, end_idx):
				prediction_tuple = sleap_predictions[l]
				if (not np.isnan(prediction_tuple[0])) and (not np.isnan(prediction_tuple[1])):
					pX += prediction_tuple[0]
					pY += prediction_tuple[1]
					points += 1.

			if points > 0:
				distances.append(np.sqrt(np.power(float(row.cX) - (pX / points), 2) + np.power(float(row.cY) - (pY / points), 2)))

		if len(distances) > 0:
			minimum_distance = np.min(distances)
			if minimum_distance < maximum_allowable_matching_distance:
				closest_instance = current_frame_instances[np.argmin(distances)]
				tag_index = np.searchsorted(tags, int(row.Tag))
				track_index = np.searchsorted(unique_tracks, int(closest_instance[4]))
				track_identity_votes[track_index, tag_index] += 1. / minimum_distance
	
	# Normalize votes
	# There are better ways to do this, but for now favor simplicity above all
	# for j in range(0, len(track_identity_votes)):
	# 	track_sum = np.sum(track_identity_votes[j])
	# 	if track_sum > 0:
	# 		track_identity_votes[j] = [k / track_sum for k in track_identity_votes[j]]

	# np.savetxt('d:\\votes.csv', track_identity_votes, delimiter = ',')

	pairings = []
	for j in range(0, len(track_identity_votes)):
		selected_track = unique_tracks[j]
		if np.max(track_identity_votes[j]) > 0:
			selected_tag = tags[np.argmax(track_identity_votes[j])]
			pairings.append((selected_track, selected_tag))

			print(f'Track {selected_track} was assigned tag {selected_tag}.')
		else:
			print(f'Track {selected_track} had no votes; ignoring')

	return pairings

def hungarian_matching(cleaned_aruco_csv_path: str, slp_predictions_path: str) -> list:
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	tag_detection_count = [np.sum(aruco_df.Tag == tag) for tag in tags]

	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	unique_tracks = np.sort(np.unique([j[4] for j in sleap_instances]))
	
	# Dictionary of the indices of tags
	tags_indices = {}
	for idx in range(len(tags)):
		tags_indices[tags[idx]] = idx

	print('\n')
	print('Unique tracks:              ', unique_tracks)
	print('Unique tags:                ', tags)

	last_frame_aruco = np.max(aruco_df.index)
	last_frame_sleap = sleap_instances[-1][2]
	last_frame = int(np.min([last_frame_aruco, last_frame_sleap]))
	print('Last frame with ArUco data: ', last_frame_aruco)
	print('Last frame with SLEAP data: ', last_frame_sleap)
	print('Overall last frame:         ', last_frame)
	print('\n')
	print('Starting tag - track distance computations ...')

	instances_idx = 0
	# Initialization code for last_frame_tracks
	current_frame_instances = []
	current_sleap_frame = sleap_frames[0] 
	for current_frame_idx in range(current_sleap_frame[3], current_sleap_frame[4]): # range(instance_id_start, instance_id_end)
		current_frame_instances.append(sleap_instances[current_frame_idx])

	last_frame_tracks = []
	chunk_end_frames = []
	hungarian_pairs = []

	m = Munkres()
	for row in aruco_df.itertuples():
		current_frame = row.Index

		# Break if we've reached the end
		if current_frame == last_frame:
			break
		
		# Put instances for current frame in a list
		# Technically, I don't think we need to put together this list... we could combine this loop with the next ones
		# Right now though, this just makes it easier to debug
		# TODO: combine this loop and the next ones
		current_frame_instances = []
		current_sleap_frame = sleap_frames[current_frame] 
		for current_frame_idx in range(current_sleap_frame[3], current_sleap_frame[4]): # range(instance_id_start, instance_id_end)
			current_frame_instances.append(sleap_instances[current_frame_idx])

		# Check if the tracks on this frame are the same.
		# If they change, run Hungarian algorithm on what we have so far, and start a new chunk
		this_frame_tracks = []
		for instance in current_frame_instances:
			this_frame_tracks.append(instance[4]) # track

		this_frame_tracks = np.sort(this_frame_tracks)

		if not np.array_equal(this_frame_tracks, last_frame_tracks):
			if len(last_frame_tracks) == 0:
				print('\n' + '=' * 80)
				print('First frame.  Created fresh cost matrix.')
				# Create new cost matrix
				mini_cost_matrix = np.zeros((len(this_frame_tracks), len(tags)))
				print(f'Matrix shape: {mini_cost_matrix.shape}')

				# Dict for the indices of the track numbers
				tracks_indices = {}
				for idx in range(len(this_frame_tracks)):
					tracks_indices[this_frame_tracks[idx]] = idx

				chunk_end_frames.append(current_frame)
				print('\n')
				print('Current frame:   ', row.Index)
				print('Tags:            ', tags)
				print('Relevant tracks: ', this_frame_tracks)
				print('\n' + '=' * 80)

			else:
				print('\n' + '=' * 80)
				print('Detected new tracks.')
				print('Previous cost matrix (raw, rounded to nearest integer):')
				print(tabulate(mini_cost_matrix.astype(int), headers = tags, tablefmt = 'fancy_grid'))
				print('\nRunning Hungarian matching algorithm on previous cost matrix.')

				# Run Hungarian algorithm
				for_hungarian = np.copy(mini_cost_matrix)
				i = 0
				for_hungarian = np.transpose(for_hungarian)
				trimmed_tags = np.copy(tags)
				# Eliminate zero-detection columns
				while i < len(for_hungarian):
					if np.sum(for_hungarian[i]) == 0:
						for_hungarian = np.delete(for_hungarian, i, 0)
						trimmed_tags = np.delete(trimmed_tags, i)
					else:
						i += 1
				# Normalize
				for j in range(len(for_hungarian)):
					row_max = np.max(for_hungarian[j])
					for k in range(len(for_hungarian[j])):
						for_hungarian[j, k] = for_hungarian[j, k] * 100. / row_max
				print('Previous cost matrix (processed, rounded to nearest integer):')
				print(tabulate(for_hungarian.astype(int), tablefmt = 'fancy_grid', headers = last_frame_tracks))
				print('Relevant tracks:            ', last_frame_tracks)
				hungarian_result = m.compute(for_hungarian)
				print('Chosen pairs:               ', hungarian_result)
				current_run_pairs = []
				for tag, track in hungarian_result:
					current_run_pairs.append((trimmed_tags[tag], last_frame_tracks[track]))
				print('Inferred tag - track pairs: ', current_run_pairs)
				hungarian_pairs.append(current_run_pairs)

				# Create new cost matrix
				print('\n\nCreating new cost matrix ...')
				current_run_tag_detections = []
				mini_cost_matrix = np.zeros((len(this_frame_tracks), len(tags)))
				print(f'Matrix shape: {mini_cost_matrix.shape}')

				# Dict for the indices of the track numbers
				tracks_indices = {}
				for idx in range(len(this_frame_tracks)):
					tracks_indices[this_frame_tracks[idx]] = idx

				chunk_end_frames.append(current_frame)
				print('\n')
				print('Current frame:   ', row.Index)
				print('Tags:            ', tags)
				print('Relevant tracks: ', this_frame_tracks)
				print('\n' + '=' * 80)


		last_frame_tracks = this_frame_tracks

		distances = []
		for instance in current_frame_instances:
			# For simplicity, find mean position of points in instance
			start_idx = instance[7]
			end_idx = instance[8]
			pX = 0.
			pY = 0.
			points = 0.
			for l in range(start_idx, end_idx):
				prediction_tuple = sleap_predictions[l]
				if (not np.isnan(prediction_tuple[0])) and (not np.isnan(prediction_tuple[1])):
					pX += prediction_tuple[0]
					pY += prediction_tuple[1]
					points += 1.

			if points > 0:
				distance = np.sqrt(np.power(float(row.cX) - (pX / points), 2) + np.power(float(row.cY) - (pY / points), 2))
				mini_cost_matrix[tracks_indices[int(instance[4])], tags_indices[int(row.Tag)]] += distance


	print('\n' + '=' * 80)
	print('Last batch to process; no new cost matrix after this.')
	print('Previous cost matrix (raw, rounded to nearest integer):')
	print(tabulate(mini_cost_matrix.astype(int), headers = tags, tablefmt = 'fancy_grid'))
	print('\nRunning Hungarian matching algorithm on previous cost matrix.')

	# Run Hungarian algorithm
	for_hungarian = np.copy(mini_cost_matrix)
	i = 0
	for_hungarian = np.transpose(for_hungarian)
	trimmed_tags = np.copy(tags)
	# Eliminate zero-detection columns
	while i < len(for_hungarian):
		if np.sum(for_hungarian[i]) == 0:
			for_hungarian = np.delete(for_hungarian, i, 0)
			trimmed_tags = np.delete(trimmed_tags, i)
		else:
			i += 1
	# Normalize
	for j in range(len(for_hungarian)):
		row_max = np.max(for_hungarian[j])
		for k in range(len(for_hungarian[j])):
			for_hungarian[j, k] = for_hungarian[j, k] * 100. / row_max

	print('Previous cost matrix (processed, rounded to nearest integer):')
	print(tabulate(for_hungarian.astype(int), tablefmt = 'fancy_grid', headers = last_frame_tracks))
	print('Relevant tracks:            ', last_frame_tracks)
	hungarian_result = m.compute(for_hungarian)
	print('Chosen pairs:               ', hungarian_result)
	current_run_pairs = []
	for tag, track in hungarian_result:
		current_run_pairs.append((trimmed_tags[tag], last_frame_tracks[track]))
	print('Inferred tag - track pairs: ', current_run_pairs)
	hungarian_pairs.append(current_run_pairs)
	
	# np.savetxt("cost_matrix.csv", track_identity_votes, delimiter=",")
	
	# Convert data into a more generalizable format
	print('\n' + '=' * 80)
	print('=' * 80)
	print('Taking pairs data into frame & track 2d array ...')
	print('Chunk breaks: ', chunk_end_frames)
	
	tag_tracks_2d_array = np.zeros((len(tags), last_frame))
	chunk_end_frames.append(last_frame)
	hungarian_pairs_list_row = 1
	for current_frame in range(last_frame):
		for tag, track in hungarian_pairs[hungarian_pairs_list_row]:
			tag_tracks_2d_array[tags_indices[tag], current_frame] = track

		if hungarian_pairs_list_row < len(hungarian_pairs) - 1:
			if current_frame == chunk_end_frames[hungarian_pairs_list_row]:
				hungarian_pairs_list_row += 1

	# print(tabulate(np.transpose(tag_tracks_2d_array), tablefmt = 'fancy_grid'))
	print('\nTracks associated with each tag:')
	associated_tracks_matrix = []
	for tag_idx in range(len(tag_tracks_2d_array)):
		associated_tracks = [f'Tag {tags[tag_idx]}']
		associated_tracks.extend(np.unique(tag_tracks_2d_array[tag_idx]))
		associated_tracks_matrix.append(np.delete(associated_tracks, 1)) # 0 points to empty entries, not tags... remove that entry.

	print(tabulate(associated_tracks_matrix, tablefmt = 'fancy_grid'))


def rolling_hungarian_matching(cleaned_aruco_csv_path: str, slp_predictions_path: str, rolling_window_size: int) -> list:
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	tag_detection_count = [np.sum(aruco_df.Tag == tag) for tag in tags]

	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	unique_tracks = np.sort(np.unique([j[4] for j in sleap_instances]))
	
	# Dictionary of the indices of tags
	tags_indices = {}
	for idx in range(len(tags)):
		tags_indices[tags[idx]] = idx

	# Dictionary of the indices of tracks
	tracks_indices = {}
	for idx in range(len(unique_tracks)):
		tracks_indices[unique_tracks[idx]] = idx

	print('\n')
	print('Unique tracks:              ', unique_tracks)
	print('Unique tags:                ', tags)

	last_frame_aruco = np.max(aruco_df.index)
	last_frame_sleap = sleap_instances[-1][2]
	last_frame = int(np.min([last_frame_aruco, last_frame_sleap]))
	print('Last frame with ArUco data: ', last_frame_aruco)
	print('Last frame with SLEAP data: ', last_frame_sleap)
	print('Overall last frame:         ', last_frame)
	print('\n')
	print('Starting tag - track distance computations ...')

	instances_idx = 0
	# Initialization code for last_frame_tracks
	current_frame_instances = []
	current_sleap_frame = sleap_frames[0] 
	for current_frame_idx in range(current_sleap_frame[3], current_sleap_frame[4]): # range(instance_id_start, instance_id_end)
		current_frame_instances.append(sleap_instances[current_frame_idx])

	last_frame_tracks = []
	chunk_end_frames = []
	hungarian_pairs = []

	tag_tracks_2d_array = np.zeros((len(tags), last_frame + 1))

	m = Munkres()

	# Prepare initial rolling data
	# We look at the last (n - 1) frames to annotate the n-th frame; the first (n - 1) frames thus have no data.

	rolling_window = []

	for row in aruco_df.itertuples():
		if row.Index < rolling_window_size:
			rolling_window.append(row)
		else:
			break

	rolling_window = pd.DataFrame(rolling_window)

	print(rolling_window)

	# When computing our cost matrix, we want to store the costs for each frame.
	# That way we can just sum over previous data, rather than recomputing costs for each new frame.
	frame_costs_dict = {}

	current_frame = rolling_window_size - 1
	while current_frame < last_frame:

		# Traverse to the first row with the next frame we'd like to see
		current_frame = aruco_df.index[np.searchsorted(aruco_df.index, current_frame, side = 'right')]

		# Erase the first part of the rolling window dataframe that's no longer in the current window
		while rolling_window.iloc[0].Index < current_frame - rolling_window_size:
			rolling_window.drop(index = rolling_window.index[0], axis = 0, inplace = True)

		# Now, append some rows to the end
		rolling_window_addition = []
		relevant_indices = range(np.searchsorted(aruco_df.index, current_frame), np.searchsorted(aruco_df.index, current_frame, side = 'right'))
		for row in aruco_df.iloc[relevant_indices].itertuples():
			rolling_window_addition.append(row)

		rolling_window_addition = pd.DataFrame(rolling_window_addition)
		rolling_window = rolling_window.append(rolling_window_addition, ignore_index = True)

		# print('Current frame: ', current_frame)
		# print(rolling_window)

		this_window_cost_matrix = np.zeros((len(unique_tracks), len(tags)))
		temp_cost_matrix = np.zeros((len(unique_tracks), len(tags)))

		# We want to add the current frame to the dict of individual frame cost matrices
		# Put instances for current frame in a list
		current_frame_instances = []
		current_sleap_frame = sleap_frames[current_frame] 
		for current_frame_idx in range(current_sleap_frame[3], current_sleap_frame[4]): # range(instance_id_start, instance_id_end)
			current_frame_instances.append(sleap_instances[current_frame_idx])

			for instance in current_frame_instances:
				# For simplicity, find mean position of points in instance
				start_idx = instance[7]
				end_idx = instance[8]
				pX = 0.
				pY = 0.
				points = 0.
				for l in range(start_idx, end_idx):
					prediction_tuple = sleap_predictions[l]
					if (not np.isnan(prediction_tuple[0])) and (not np.isnan(prediction_tuple[1])):
						pX += prediction_tuple[0]
						pY += prediction_tuple[1]
						points += 1.

				if points > 0:
					distance = np.sqrt(np.power(float(row.cX) - (pX / points), 2) + np.power(float(row.cY) - (pY / points), 2))
					temp_cost_matrix[tracks_indices[int(instance[4])], tags_indices[int(row.Tag)]] += distance
		frame_costs_dict[current_frame] = temp_cost_matrix
		# print(tabulate(temp_cost_matrix.astype(int), tablefmt = 'fancy_grid'))

		# Delete information for frame now outside of window
		if current_frame - rolling_window_size - 1 in frame_costs_dict.keys():
			del frame_costs_dict[current_frame - rolling_window_size - 1]

		for key in frame_costs_dict.keys():
			this_window_cost_matrix += frame_costs_dict[key]

		for_hungarian = np.copy(this_window_cost_matrix)
		i = 0
		trimmed_tracks = np.copy(unique_tracks)
		# Eliminate zero-detection columns
		while i < len(for_hungarian):
			if np.sum(for_hungarian[i]) == 0:
				for_hungarian = np.delete(for_hungarian, i, 0)
				trimmed_tracks = np.delete(trimmed_tracks, i)
			else:
				i += 1
		i = 0
		for_hungarian = np.transpose(for_hungarian)
		trimmed_tags = np.copy(tags)
		# Eliminate zero-detection columns
		while i < len(for_hungarian):
			if np.sum(for_hungarian[i]) == 0:
				for_hungarian = np.delete(for_hungarian, i, 0)
				trimmed_tags = np.delete(trimmed_tags, i)
			else:
				i += 1

		# Normalize
		for j in range(len(for_hungarian)):
			row_max = np.max(for_hungarian[j])
			for k in range(len(for_hungarian[j])):
				for_hungarian[j, k] = for_hungarian[j, k] * 100. / row_max

		# Disallow zero entries
		for_hungarian = np.where(for_hungarian == 0., 999., for_hungarian)

		print('Previous cost matrix (processed, rounded to nearest integer, trasposed for display purposes):')
		print(tabulate(for_hungarian.astype(int), tablefmt = 'fancy_grid'))
		hungarian_result = m.compute(for_hungarian)
		print('Chosen pairs:               ', hungarian_result)
		hungarian_pairs = []
		for tag, track in hungarian_result:
			hungarian_pairs.append((trimmed_tags[tag], trimmed_tracks[track]))
		print('Inferred tag - track pairs: ', hungarian_pairs)

		for tag, track in hungarian_pairs:
			tag_tracks_2d_array[tags_indices[tag], current_frame] = track

		idx = 0
		
	for current_frame in range(last_frame):
		for idx in range(len(tag_tracks_2d_array[:, current_frame])):
			if tag_tracks_2d_array[idx, current_frame] == 0 and current_frame > 0:
				tag_tracks_2d_array[idx, current_frame] = tag_tracks_2d_array[idx, current_frame - 1]

	return tag_tracks_2d_array


def hungarian_annotate_video_sleap_aruco_pairings(video_path: str, video_output_path: str, cleaned_aruco_csv_path: str, slp_predictions_path: str,\
 pairings: np.ndarray, frames_to_annotate: list, tl_coords: tuple[int, int], square_dim: int):
	'''
	annotate video with pairings of aruco tags and sleap tracks
	output is .avi
	'''
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))

	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])

	video_data = cv2.VideoCapture(video_path)
	fps = video_data.get(cv2.CAP_PROP_FPS)
	font = cv2.FONT_HERSHEY_SIMPLEX

	out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (800, 800))#(square_dim, square_dim))

	success, image = video_data.read()

	font = cv2.FONT_HERSHEY_SIMPLEX

	# ArUco persistent plot
	last_seen = np.zeros((2, len(tags)), dtype = int)

	current_frame_idx = 0
	next_frame_idx = 0
	for frame in frames_to_annotate:
		video_data.set(1, frame)
		image = image[tl_coords[0]:tl_coords[0] + square_dim, tl_coords[1]:tl_coords[1] + square_dim]
		success, image = video_data.read()
		# Find starting point in .slp instances data
		nth_inst_tuple = sleap_instances[current_frame_idx]
		while nth_inst_tuple[2] != frame + 1:  # frame_id
			next_frame_idx += 1
			nth_inst_tuple = sleap_instances[next_frame_idx]

		for idx in range(current_frame_idx, next_frame_idx):
			nth_inst_tuple = sleap_instances[idx]
			prediction_start_idx = nth_inst_tuple[7]
			prediction_end_idx = nth_inst_tuple[8]
			prediction_coords = [[],[]] # x, y
			for pred_idx in range(prediction_start_idx, prediction_end_idx):
				prediction_tuple = sleap_predictions[pred_idx]
				prediction_coords[0].append(float(prediction_tuple[0]))
				prediction_coords[1].append(float(prediction_tuple[1]))
				try:
					image = cv2.circle(image, (int(round(float(prediction_tuple[0]))), int(round(float(prediction_tuple[1])))), 5, (0, 255, 0), 2)
				except:
					print(prediction_tuple)

			try:
				mean_coords = (int(np.round(np.mean(prediction_coords[0]))), int(np.round(np.mean(prediction_coords[1]))))
				image = cv2.circle(image, mean_coords, 50, (255, 0, 0), 2)
			except:
				print(mean_coords)

			current_track = nth_inst_tuple[4]
			current_tag = '?'
			for tag_track_idx in range(len(pairings[:, frame])):
				if pairings[tag_track_idx, frame] == current_track:
					current_tag = tags[tag_track_idx]
			cv2.putText(image, str(current_tag), (mean_coords[0], mean_coords[1] - 50), font, 2, (255, 0, 0), 2)
			cv2.putText(image, str(current_track), (mean_coords[0], mean_coords[1] + 75), font, 1, (255, 0, 0), 2)

		# Persistent ArUco location plot
		if frame in aruco_df.index:
			for row in aruco_df[aruco_df.index == frame].itertuples():
				last_seen[0, np.searchsorted(tags, row.Tag)] = int(round(row.cX))
				last_seen[1, np.searchsorted(tags, row.Tag)] = int(round(row.cY))

		for k in range(0,len(tags)):
			if last_seen[0, k] > 0 and last_seen[1, k] > 0:
				image = cv2.circle(image, (last_seen[0, k], last_seen[1, k]), 5, (0, 0, 255), 2)
				cv2.putText(image, str(tags[k]), (last_seen[0, k], last_seen[1, k]), font, 2, (0, 0, 255), 2)



		image = cv2.resize(image, (800,800))
		out.write(image)
		# cv2.imshow('', image)
		# cv2.waitKey(1)

		current_frame_idx = next_frame_idx

	# Do a bit of cleanup
	video_data.release()
	out.release()
	cv2.destroyAllWindows()


def annotate_video_sleap_aruco_pairings(video_path: str, video_output_path: str, cleaned_aruco_csv_path: str, slp_predictions_path: str,\
 pairings: list, frames_to_annotate: list, tl_coords: tuple[int, int], square_dim: int):
	'''
	annotate video with pairings of aruco tags and sleap tracks
	output is .avi
	'''
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])

	video_data = cv2.VideoCapture(video_path)
	fps = video_data.get(cv2.CAP_PROP_FPS)
	font = cv2.FONT_HERSHEY_SIMPLEX
	out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (800, 800))#(square_dim, square_dim))

	success, image = video_data.read()
	font = cv2.FONT_HERSHEY_SIMPLEX

	# ArUco persistent plot
	last_seen = np.zeros((2, len(tags)), dtype = int)

	current_frame_idx = 0
	next_frame_idx = 0
	for frame in frames_to_annotate:
		video_data.set(1, frame)
		image = image[tl_coords[0]:tl_coords[0] + square_dim, tl_coords[1]:tl_coords[1] + square_dim]
		success, image = video_data.read()
		# Find starting point in .slp instances data
		nth_inst_tuple = sleap_instances[current_frame_idx]
		while nth_inst_tuple[2] != frame + 1:  # frame_id
			next_frame_idx += 1
			nth_inst_tuple = sleap_instances[next_frame_idx]

		for idx in range(current_frame_idx, next_frame_idx):
			nth_inst_tuple = sleap_instances[idx]
			prediction_start_idx = nth_inst_tuple[7]
			prediction_end_idx = nth_inst_tuple[8]
			prediction_coords = [[],[]] # x, y
			for pred_idx in range(prediction_start_idx, prediction_end_idx):
				prediction_tuple = sleap_predictions[pred_idx]
				prediction_coords[0].append(float(prediction_tuple[0]))
				prediction_coords[1].append(float(prediction_tuple[1]))
				try:
					image = cv2.circle(image, (int(round(float(prediction_tuple[0]))), int(round(float(prediction_tuple[1])))), 5, (0, 255, 0), 2)
				except:
					print(prediction_tuple)
			try:
				mean_coords = (int(np.round(np.mean(prediction_coords[0]))), int(np.round(np.mean(prediction_coords[1]))))
				image = cv2.circle(image, mean_coords, 50, (255, 0, 0), 2)
				current_track = nth_inst_tuple[4]
				current_tag = '?'
				for pair in pairings:
					if pair[0] == current_track:
						current_tag = pair[1]
				cv2.putText(image, str(current_tag), (mean_coords[0], mean_coords[1] - 50), font, 2, (255, 0, 0), 2)
				cv2.putText(image, str(current_track), (mean_coords[0], mean_coords[1] + 75), font, 1, (255, 0, 0), 2)
			except:
				print(mean_coords)

		# Persistent ArUco location plot
		if frame in aruco_df.index:
			for row in aruco_df[aruco_df.index == frame].itertuples():
				last_seen[0, np.searchsorted(tags, row.Tag)] = int(round(row.cX))
				last_seen[1, np.searchsorted(tags, row.Tag)] = int(round(row.cY))

		for k in range(0,len(tags)):
			if last_seen[0, k] > 0 and last_seen[1, k] > 0:
				image = cv2.circle(image, (last_seen[0, k], last_seen[1, k]), 5, (0, 0, 255), 2)
				cv2.putText(image, str(tags[k]), (last_seen[0, k], last_seen[1, k]), font, 2, (0, 0, 255), 2)



		image = cv2.resize(image, (800,800))
		out.write(image)
		# cv2.imshow('', image)
		# cv2.waitKey(1)

		current_frame_idx = next_frame_idx

	# Do a bit of cleanup
	video_data.release()
	out.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# pairings = proximity_matching('d:\\sleap-tigergpu\\20210506_12h_Tracked_DefaultArUcoParams.csv', 'd:\\20210505_run003_00000000.mp4.predictions.slp', 50)
	# pairings = hungarian_matching('d:\\sleap-tigergpu\\20210506_12h_Tracked_DefaultArUcoParams.csv', 'd:\\20210505_run003_00000000.mp4.predictions.slp')
	pairings = rolling_hungarian_matching('d:\\sleap-tigergpu\\20210506_12h_Tracked_DefaultArUcoParams.csv', 'd:\\20210505_run003_00000000.mp4.predictions.slp', 300)
	np.savetxt("aruco_sleap_matching_output.csv", pairings, delimiter = ",")
	hungarian_annotate_video_sleap_aruco_pairings('d:\\sleap-tigergpu\\20210505_run003_00000000.mp4', 'd:\\sleap-tigergpu\\matching_test_ecldn_weight.avi', 'd:\\sleap-tigergpu\\20210506_12h_Tracked_DefaultArUcoParams.csv', 'd:\\sleap-tigergpu\\20210505_run003_00000000_48_CI_filters.avi.predictions.slp', pairings, range(300, 10000), (0, 0), 2063)


	# results_df = ArUco_cleaner('d:\\sleap-tigergpu\\20210505_run003_00000000.mp4', range(0, 24000), (0, 0), 2063, 16)
	# results_df.to_csv("aruco_clean_results.csv", index = False)
	# tags = awpd.find_tags_fast(results_df)
	# aruco_dict_by_tag = awpd.sort_for_individual_tags(results_df, tags)
	# awpd.print_by_tag_data(tags, aruco_dict_by_tag)


	# aruco_df_by_tag.to_csv("results_by_tag.csv", index = False)
	# 'd:\\sleap-tigergpu\\20210505_run003_00000000.avi.predictions.slp'