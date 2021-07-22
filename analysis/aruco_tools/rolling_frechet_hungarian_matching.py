import sys
import math
import time
import cv2
import numpy as np
import pandas as pd
import h5py
import skvideo.io
from tabulate import tabulate
from munkres import Munkres

import aruco_utils_pd as awpd
import discrete_frechet as frechet


def sleap_reader(slp_predictions_path: str) -> h5py.File:
	f = h5py.File(slp_predictions_path, 'r')
	return f


def sigmoid(input_value: float, center: float, slope: float):
	'''
	sigmoid = 1 / (1 + exp[-slope * {input - centur}])
	'''
	return 1 / (1. + np.exp(-slope * (input_value - center)))


def rolling_hungarian_matching_frechet(cleaned_aruco_csv_path: str, slp_predictions_path: str, start_end_frames_assignment: tuple[int, int], last_frame_in_video: int,\
 half_rolling_window_size: int, maximum_hop_length: int, minimum_sleap_score: float, min_tag_detection_freq: float, min_track_detection_freq: float,\
 window_min_tag_detection_freq: float, window_min_track_detection_freq: float, pairing_threshold: float,\
  enhanced_output: bool = True) -> list:
	# Load relevant data into RAM =============================================================================================================================

	# ArUco data:
	# We load this in as a dataframe, and keep a separate numpy array of the tag numbers for quick reference
	# Furthermore, we sort the dataframe into a dict of dataframes, each holding the data for only one individual tag
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	aruco_dict_by_tag = awpd.sort_for_individual_tags(aruco_df, tags)

	# SLEAP file loading
	# We load in parts of the file as arrays of tuples
	# To get a good idea of what's going on, it's good to use MATLAB to visualize the structure of the SLEAP file: h5disp({SLEAP file path})
	# Although it ends with .slp, the SLEAP file is an h5 file
	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	# Not sure why, but we get a 'track -1'... get rid of it!
	unique_tracks = np.sort(np.unique([int(j[4]) for j in sleap_instances]))
	if unique_tracks[0] == -1:
		unique_tracks = unique_tracks[1:-1]

	# Interpolation ===========================================================================================================================================

	start_end_frame = (np.maximum(0, start_end_frames_assignment[0] - half_rolling_window_size), np.minimum(last_frame_in_video, start_end_frames_assignment[1] + half_rolling_window_size))

	# So far, we have a dict with dataframes for each tag
	# The frames where the tag wasn't detected are simply not represented in the dataframe.
	# In the next step, we create a much bigger dataframe with NaNs filling gaps less than the parameter maximum_hop_length
	# Then, we can quickly fill these gaps with the pandas.interpolate() function.
	# At the same time, we also shed data from frames that are outside of the computation range.
	# Results are again stored as a dict of dataframes
	print('[ArUco Interpolation] Started')
	aruco_interpolation_start = time.perf_counter()

	# For each tag
	killed_tags = []
	for tag in tags:
		# Kill dataframes with not enough data
		if len(aruco_dict_by_tag[tag].index) < 2 * half_rolling_window_size * min_tag_detection_freq:
			del aruco_dict_by_tag[tag]
			tags = np.setdiff1d(tags, [tag])
			killed_tags.append(tag)
			continue

		# Initialize empty list: we will convert this to a dataframe later on
		# Appending to a list is much cheaper than appending to a dataframe.
		NaN_filled_data = []

		#
		previous_frame_number_with_data = -1

		# Iterate through the rows of the dataframe for this particular tag
		for row in aruco_dict_by_tag[tag].itertuples():
			# If the current row corresponds to a frame within the assigned range
			# If the end of the selection range corresponds to a gap in the data, we still want to interpolate up to the very end
			# No need to interpolate if the previous frame with data was the frame one before the current one
			# Also, no need to interpolate if the gap is too long, as specified by maximum_hop_length
			# row.Index is the current frame number with data that we're looking at
			if (row.Index >= start_end_frame[0] and previous_frame_number_with_data <= start_end_frame[1]):
				gap_length_in_frames = row.Index - previous_frame_number_with_data
				if (gap_length_in_frames > 1 and gap_length_in_frames <= maximum_hop_length):
					# Interpolate through the gap.
					# This looks more complicated then it actually is; it just deals with edge cases so we interpolate to the edges of the relevant frame range when permitted to do so
					# We might have up to maximum_hop_length extra values at the end of the dataframe, but it's ok.
					for frame_number in range(np.maximum(previous_frame_number_with_data + 1, start_end_frame[0]), row.Index):
						# Append NaNs for the missing frames
						NaN_filled_data.append((frame_number, math.nan, math.nan))

			# Append the most recent row
			NaN_filled_data.append((row.Index, row.cX, row.cY))

			if row.Index > start_end_frame[1]:
				break

			previous_frame_number_with_data = row.Index

		# Convert to dataframe and slot back into the dict
		aruco_dict_by_tag[tag] = pd.DataFrame(NaN_filled_data, columns = ['Frame', 'cX', 'cY'])

		# Interpolate!!
		aruco_dict_by_tag[tag].interpolate(inplace = True, limit_direction = 'both')

	if enhanced_output:
		for tag in tags:
			print(f'\n\nTag: {tag}')
			print(aruco_dict_by_tag[tag])
	
	aruco_interpolation_end = time.perf_counter()
	print(f'Killed tags: {killed_tags}')
	print(f'[ArUco Interpolation] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0]) / float(aruco_interpolation_end - aruco_interpolation_start), 2)}')


	# Now, let's put the relevant SLEAP tracks into the same data structure: a dict containing interpolated coords for each track
	print('[SLEAP Data Restructuring] Started')
	sleap_interpolation_start = time.perf_counter()
	sleap_predictions_dict_by_track = {}
	for track in unique_tracks:
		sleap_predictions_dict_by_track[track] = []

	for frame_number in range(start_end_frame[0], start_end_frame[1]):
		iter_sleap_frame = sleap_frames[frame_number] 
		for iter_frame_idx in range(iter_sleap_frame[3], iter_sleap_frame[4]): # range(instance_id_start, instance_id_end)
			current_instance = sleap_instances[iter_frame_idx]
			prediction_index = current_instance[7] # Member 'point_id_start':  H5T_STD_U64LE (uint64)
			track_number = current_instance[4] # Member 'track':  H5T_STD_I32LE (int32)
			prediction = sleap_predictions[prediction_index]
			if prediction[4] >= minimum_sleap_score: # Member 'score':  H5T_IEEE_F64LE (double)
				# if prediction[2] == 1 and prediction[3] == 1: # Member 'visible':  H5T_ENUM, Member 'complete':  H5T_ENUM
				sleap_predictions_dict_by_track[track_number].append((frame_number, float(prediction[0]), float(prediction[1])))

	killed_tracks = []
	for track in unique_tracks:
		sleap_predictions_dict_by_track[track] = pd.DataFrame(sleap_predictions_dict_by_track[track], columns = ['Frame', 'cX', 'cY'])

		# Kill dataframes with not enough data
		if len(sleap_predictions_dict_by_track[track].index) < 2 * half_rolling_window_size * min_track_detection_freq:
			del sleap_predictions_dict_by_track[track]
			unique_tracks = np.setdiff1d(unique_tracks, [track])
			killed_tracks.append(track)

	if enhanced_output:
		for track in unique_tracks:
			print(f'\n\nTrack: {track}')
			print(sleap_predictions_dict_by_track[track])

	sleap_interpolation_end = time.perf_counter()
	print(f'Killed tracks: {killed_tracks}')
	print(f'[SLEAP Data Restructuring] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0]) / float(sleap_interpolation_end - sleap_interpolation_start), 2)}')


	# Rolling window ==========================================================================================================================================
	# Initialize Frechet matching class:
	fast_frechet = frechet.FastDiscreteFrechetMatrix(frechet.euclidean)
	# Initialize Hungarian algorithm clss:
	m = Munkres()
	# Array to store results:
	tag_tracks_2d_array = np.zeros((1 + len(tags), 1 + start_end_frames_assignment[1] - start_end_frames_assignment[0]))
	tag_tracks_2d_array[:, 0] = np.concatenate(([0], tags))
	tag_tracks_2d_array[0, :] = np.concatenate(([-1], np.arange(start_end_frames_assignment[0], start_end_frames_assignment[1])))

	# Dictionary of the indices of tags
	untrimmed_tags_indices = {}
	for idx in range(len(tags)):
		untrimmed_tags_indices[tags[idx]] = idx

	print('[Rolling Window Hungarian Matching with Fréchet Distance Cost Matrices] Started')
	RWHMwFDCM_start = time.perf_counter()
	for center_of_window_frame in awpd.progressBar(range(start_end_frames_assignment[0], start_end_frames_assignment[1])):
		if enhanced_output:
			print('\n\n' + '=' * 80)
			print(f'Frame (center of window): {center_of_window_frame}\n')

		tags_window_dict = {}
		trimmed_tags = np.copy(tags)
		for tag in tags:
			tags_window_dict[tag] = aruco_dict_by_tag[tag].loc[center_of_window_frame - half_rolling_window_size : center_of_window_frame + half_rolling_window_size]
			if len(tags_window_dict[tag].index) < 2 * half_rolling_window_size * window_min_tag_detection_freq:
				del tags_window_dict[tag]
				trimmed_tags = np.setdiff1d(trimmed_tags, [tag])
				if enhanced_output:
					print(f'\nKilled tag #{tag}')
			elif enhanced_output:
				print('\n\n')
				print(f'Tag #{tag}:')
				if enhanced_output:
					print(tags_window_dict[tag])

		tracks_window_dict = {}
		trimmed_tracks = np.copy(unique_tracks)
		for track in unique_tracks:
			tracks_window_dict[track] = sleap_predictions_dict_by_track[track].loc[center_of_window_frame - half_rolling_window_size : center_of_window_frame + half_rolling_window_size]
			if len(tracks_window_dict[track].index) < 2 * half_rolling_window_size * window_min_track_detection_freq:
				del tracks_window_dict[track]
				trimmed_tracks = np.setdiff1d(trimmed_tracks, [track])
				if enhanced_output:
					print(f'\nKilled track #{track}')
			elif enhanced_output:
				print('\n\n')
				print(f'Track #{track}:')
				if enhanced_output:
					print(tracks_window_dict[track])

		# Initialize this frame's cost matrix and dictionaries for indices
		cost_matrix = np.zeros((len(trimmed_tags), len(trimmed_tracks)))

		# Dictionary of the indices of tags
		tags_indices = {}
		for idx in range(len(trimmed_tags)):
			tags_indices[trimmed_tags[idx]] = idx

		# Dictionary of the indices of tracks
		tracks_indices = {}
		for idx in range(len(trimmed_tracks)):
			tracks_indices[trimmed_tracks[idx]] = idx

		# Iterate through every combination of tags and tracks
		for tag in trimmed_tags:
			for track in trimmed_tracks:
				current_tag_path = np.transpose(np.stack((tags_window_dict[tag]['cX'].values, tags_window_dict[tag]['cY'].values)))
				current_track_path = np.transpose(np.stack((tracks_window_dict[track]['cX'].values, tracks_window_dict[track]['cY'].values)))

				# Calculate the Fréchet distance between ArUco tag and SLEAP prediction paths
				frechet_distance = fast_frechet.distance(current_tag_path, current_track_path)

				# Calculate cost
				cost = frechet_distance

				cost_matrix[tags_indices[tag], tracks_indices[track]] = cost

		if len(trimmed_tracks) < len(trimmed_tags):
			cost_matrix = np.transpose(cost_matrix)
			hungarian_result = m.compute(np.copy(cost_matrix))
			hungarian_pairs = []
			for track, tag in hungarian_result:
				hungarian_pairs.append((trimmed_tags[tag], trimmed_tracks[track]))
			cost_matrix = np.transpose(cost_matrix)
		else:
			hungarian_result = m.compute(np.copy(cost_matrix))
			hungarian_pairs = []
			for tag, track in hungarian_result:
				hungarian_pairs.append((trimmed_tags[tag], trimmed_tracks[track]))

		# Assignment threshold
		assigned = []
		for tag, track in hungarian_pairs:
			if cost_matrix[tags_indices[tag], tracks_indices[track]] < pairing_threshold:
				assigned.append(tag)
				tag_tracks_2d_array[1 + untrimmed_tags_indices[tag], 1 + center_of_window_frame - start_end_frames_assignment[0]] = track

		cost_matrix = cost_matrix / 1000.
		if enhanced_output:
			print('\n\n' + '=' * 80)
			print('Current frame:               ', center_of_window_frame)
			print('Chosen pairs:                ', hungarian_result)
			print('Inferred tag - track pairs:  ', hungarian_pairs)
			print(tabulate(np.round(cost_matrix, 1), tablefmt = 'pretty', headers = trimmed_tracks))


	RWHMwFDCM_end = time.perf_counter()
	print(f'[Rolling Window Hungarian Matching with Fréchet Distance Cost Matrices] Ended, FPS: {round(float(start_end_frames_assignment[1] - start_end_frames_assignment[0]) / float(RWHMwFDCM_end - RWHMwFDCM_start), 2)}')

	# Inherit tracks
	for current_frame in range(1 + start_end_frames_assignment[0], start_end_frames_assignment[1] - start_end_frames_assignment[0]):
		for idx in range(1, len(tag_tracks_2d_array[:, current_frame])):
			if tag_tracks_2d_array[idx, current_frame] == 0 and current_frame > 0:
				tag_tracks_2d_array[idx, current_frame] = tag_tracks_2d_array[idx, current_frame - 1]

	return tag_tracks_2d_array


def hungarian_annotate_video_sleap_aruco_pairings(video_path: str, video_output_path: str, aruco_csv_path: str, slp_predictions_path: str,\
 pairings: np.ndarray, start_end_frame: tuple[int, int], display_output_on_screen: bool = False) -> None:
	'''
	annotate video with pairings of aruco tags and sleap tracks
	output is .avi
	'''
	aruco_df = awpd.load_into_pd_dataframe(aruco_csv_path)
	tags = pairings[1 : -1, 0]

	pairings = pairings[1 : -1, 1 : -1].astype(int)

	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	video_data = cv2.VideoCapture(video_path)
	fps = video_data.get(cv2.CAP_PROP_FPS)
	font = cv2.FONT_HERSHEY_SIMPLEX

	# Below code heavily based on SLEAP (sleap.io.videowriter.py)
	fps = str(fps)
	crf = 21
	preset = 'superfast'
	writer = skvideo.io.FFmpegWriter(
				video_output_path,
				inputdict={
					"-r": fps,
				},
				outputdict={
					"-c:v": "libx264",
					"-preset": preset,
					"-framerate": fps,
					"-crf": str(crf),
					"-pix_fmt": "yuv420p",
				}, #verbosity = 1
			)

	success, image = video_data.read()

	font = cv2.FONT_HERSHEY_SIMPLEX

	# ArUco persistent plot
	last_seen = np.zeros((3, len(tags)), dtype = int)

	current_frame_idx = 0
	next_frame_idx = 0
	video_data.set(1, start_end_frame[0])
	success = True
	frame = start_end_frame[0]
	errors = 0
	while success:
		# print('\n' + '=' * 80)
		# print(f'Frame {frame}')
		success, image = video_data.read()

		current_sleap_frame = sleap_frames[frame]
		current_frame_idx = current_sleap_frame[3]
		next_frame_idx = current_sleap_frame[4]
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
					# print(prediction_tuple)
					errors += 1

			try:
				prediction_tuple = sleap_predictions[prediction_start_idx] # start_idx corresponds to the tag
				pX = int(round(prediction_tuple[0]))
				pY = int(round(prediction_tuple[1]))
				image = cv2.circle(image, (pX, pY), 75, (255, 0, 0), 2)
				current_track = int(nth_inst_tuple[4])
				current_tag = '?'
				for tag_track_idx in range(len(pairings[:, frame])):
					if pairings[tag_track_idx, frame] == current_track:
						current_tag = int(tags[tag_track_idx])
				cv2.putText(image, str(current_tag), (pX, pY - 75), font, 4, (255, 0, 0), 2)
				cv2.putText(image, str(current_track), (pX, pY + 100), font, 2, (255, 0, 0), 2)
			except:
				# print(f'Failure on frame {frame}')
				errors += 1

		# Persistent ArUco location plot
		if frame in aruco_df.index:
			for row in aruco_df[aruco_df.index == frame].itertuples():
				if row.Tag in tags:
					last_seen[0, np.searchsorted(tags, row.Tag)] = int(round(row.cX))
					last_seen[1, np.searchsorted(tags, row.Tag)] = int(round(row.cY))
					last_seen[2, np.searchsorted(tags, row.Tag)] = float(row.Theta)

		for k in range(0,len(tags)):
			if last_seen[0, k] > 0 and last_seen[1, k] > 0:
				image = cv2.circle(image, (last_seen[0, k], last_seen[1, k]), 5, (0, 0, 255), 2)
				Theta = last_seen[2, k]
				cX = last_seen[0, k]
				cY = last_seen[1, k]
				image = cv2.line(image, (int(round(cX - 50 * np.cos(Theta))), int(round(cY - 50 * np.sin(Theta)))), (int(round(cX + 150 * np.cos(Theta))), int(round(cY  + 150 * np.sin(Theta)))), (0, 0, 255), 2)
				cv2.putText(image, str(int(tags[k])), (last_seen[0, k], last_seen[1, k]), font, 2, (0, 0, 255), 2)


		writer.writeFrame(image)
		
		if display_output_on_screen:
			image = cv2.resize(image, (800,800))
			cv2.imshow('', image)
			cv2.waitKey(1)

		current_frame_idx = next_frame_idx
		frame += 1

		if frame == start_end_frame[1]:
			break

	# Do a bit of cleanup
	video_data.release()
	writer.close()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	start_frame = 0
	end_frame = 600
	half_rolling_window_size = 20
	maximum_hop_length = 10
	minimum_sleap_score = 0.1
	min_tag_detection_freq = 0.8
	min_track_detection_freq = 0.8
	window_min_tag_detection_freq = 0.5
	window_min_track_detection_freq = 0.5
	pairing_threshold = 25.
	pairings = rolling_hungarian_matching_frechet('d:\\20210715_run001_00000000_cut_aruco_annotated.csv', 'd:\\20210715_run001_00000000_cut.mp4.predictions.slp', (start_frame, end_frame), 1200, half_rolling_window_size, maximum_hop_length, minimum_sleap_score, min_tag_detection_freq, min_track_detection_freq, window_min_tag_detection_freq, window_min_track_detection_freq, pairing_threshold, enhanced_output = False)
	np.savetxt("aruco_sleap_matching_output.csv", pairings, delimiter = ",")

	hungarian_annotate_video_sleap_aruco_pairings('d:\\20210715_run001_00000000_cut.mp4', 'd:\\frechet_test.mp4', 'd:\\20210715_run001_00000000_cut_aruco_annotated.csv', 'd:\\20210715_run001_00000000_cut.mp4.predictions.slp', pairings, (start_frame, end_frame), display_output_on_screen = True)