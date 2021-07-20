# Scroll to the bottom to see detailed comments and the entire pipeline condensed into a single block of code.
'''
********************************************************************************
********************************************************************************
************************************/(#/****************************************
**************************************/%(*//*****//*****************************
***************************************/(%///////**////**//*********************
********************************//**//////%#//////////////////*///**************
*********************************//////////(%((/((//////////*****/**************
*****************************/////((((((((((((#%%(((((////////***//(((/*********
**********************///////////(%&&(((((%&&&&&&&&&&&&&%#%%%#((///**/**********
*********************////////////(%&&%(#%&@@@@&&&@@@@&%((////////////***********
****************/****//////////(((((#/**/#%&&&&&&&@@@&&&#((///////////**********
**********/*****//////////////(((((//(&&&#/**(&&%%%%&&&&#((((////////***********
*********/**///////////////////((/*/%&%//(%&&%/***(#%%%%&&&&%(///////*/**/******
********/////////(((((((((((((((**#&&&%/*****(%&&#*,/##%%%#((((//////***********
******//////(%%&&&&&&&&&#(((((/*(&&&&&&&&(///(&&(**(%#((((((////////*/**********
******//(###%%%%%##%&&&&&&&%#(*,**(%&&&&&&&&&&#*,*%&&#((((////////////**********
***//(((/////////((((((((#%&&&&%(/***/(%&&&&%(,*(&&&%#(((((((///////////********
***/**//////////////((((((###%&&%%&&%#****//*,/%&%&%###((((((((////////*/*******
***//*//*//////////((((#&&&&&&&%%&&&&&&&&%(//(%%%%%####((((((((((///////********
*****//////////////((#&&%##%&&&%%&&&&@&&@@&&&%&%%&@&&&&&&&&&&&&&%#(//////*******
*//**/////////////((%&&&##%&&&%%&&%&&@@&&%%%&&%%&&%%%%%#((#%%&&%&&&%#///********
***/**////////////(%&&&%#%@@&&%&&%%&&&&%%%%&%%&&&@@&&#(((((((((((#%%%%(//**/****
*****////////////(#%&&&%#&&&&&&&&&&&&&%&%&&%%&&&%%%&&%((((((((////////(#(/******
*****////////////(#&&&&#%&&&&&&%&&&%%&&&%&%%&&&&%#&@&%#((////////////////(///***
*******//////////(#&&#(#&&&&&%%%&&&&&%%&%%&&&&&&%&&&&#((///////////////*********
***/*/*/////////(%&#((((%%&#&&&&&&&&&%%%%%&&&&&&&&&&%((//////////////////*******
******//////////##(/((((#%%%&&&&&&&&%&&&&&&&&&&%&&&%#(///////////////********/**
*******/*///////%(/////((%#%&&&%%%%&&&&&&&&&&&%#&&%#((//////////******////******
*******//**/////#(///////##%%%%%%&&&&&&&&&&&%#((%&#((///////////*///*****/******
*******/*/***///((///////(#%%&&&&&&&&&&%&&&#((((%#(/////////////***/*****/***//*
**********//////*//////////(((##%%%%%%&%%##((((#((///////////**/////*/////***/**
**********/****//////////////((((((((((((((/((#(///////////////////*////////////
'''

import sys
import cv2
import skvideo.io
import numpy as np
import pandas as pd
import time
import h5py
from datetime import datetime
from threading import Thread
from contextlib import redirect_stdout
from tabulate import tabulate
from munkres import Munkres

import aruco_utils_pd as awpd

def threaded_aruco_annotate_video(video_path: str, video_output_path: str, csv_output_path: str, tl_coords: tuple[int,int], dimension: int, start_end_frames: tuple[int, int], annotate_video: bool = False, display_output_on_screen: bool = False, display_dim: tuple[int, int] = (800, 800)) -> None:
	# define names of a few possible ArUco tag OpenCV supports
	ARUCO_DICT = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	}

	video_getter = VideoGet(video_path, start_end_frames[0]).start()

	tagset_name = "DICT_4X4_50"
	print("[INFO] detecting '{}' tags...".format(tagset_name))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tagset_name])
	arucoParams = cv2.aruco.DetectorParameters_create()

	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	arucoParams.adaptiveThreshWinSizeStep = 3
	arucoParams.adaptiveThreshWinSizeMin = 3
	arucoParams.adaptiveThreshWinSizeMax = 30

	arucoParams.adaptiveThreshConstant = 12

	arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	arucoParams.errorCorrectionRate = 1.

	print("[INFO] starting video stream...")
	
	if annotate_video:
		vs = cv2.VideoCapture(video_path)
		fps = vs.get(cv2.CAP_PROP_FPS)
		vs.release()

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

	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY', 'Theta'])

	# loop over the frames from the video stream

	start_time = datetime.now()

	font = cv2.FONT_HERSHEY_SIMPLEX

	# Top left x and y for  cropping
	tlx = tl_coords[0]
	tly = tl_coords[1]

	# Assuming this is a square!
	dimension;

	i = start_end_frames[0]
	detected = 0

	while not video_getter.stopped:
		frame = video_getter.frame
		
		# Setting dimension to a negative number disables cropping functionality
		if dimension > 0:
			frame = frame[tlx:tlx + dimension, tly:tly + dimension]

		# detect ArUco markers in the input frame
		(corners, ids, rejected) = cv2.aruco.detectMarkers(
			frame, arucoDict, parameters=arucoParams)

		detected += len(corners)
		if len(corners) > 0:
			len(corners)
			for (markerCorner, markerID) in zip(corners, ids):
				corners = markerCorner.reshape((4, 2))
				(topLeft, topRight, bottomRight, bottomLeft) = corners

				# convert each of the (x, y)-coordinate pairs to integers
				topRight = (topRight[0], topRight[1])
				bottomRight = (bottomRight[0], bottomRight[1])
				bottomLeft = (bottomLeft[0], bottomLeft[1])
				topLeft = (topLeft[0], topLeft[1])
				
				# Calculate centroid x a
				cX = (topLeft[0] + bottomRight[0]) / 2.0
				cY = (topLeft[1] + bottomRight[1]) / 2.0

				Theta = np.arctan2(topRight[1] - bottomLeft[1], topRight[0] - bottomLeft[0])

				# 'Frame', 'Tag', 'cX','cY', 'Theta'
				results_df.loc[len(results_df)] = [int(i), int(markerID[0]), cX, cY, Theta]

				if annotate_video:
					frame = cv2.circle(frame, (int(round(cX)), int(round(cY))), 100, (255, 0, 0), 2)
					frame = cv2.line(frame, (int(round(cX - 50 * np.cos(Theta))), int(round(cY - 50 * np.sin(Theta)))), (int(round(cX + 150 * np.cos(Theta))), int(round(cY  + 150 * np.sin(Theta)))), (255, 0, 0), 2)
					cv2.putText(frame, str(int(markerID[0])), (int(round(cX)), int(round(cY)) - 100), font, 2, (255, 0, 0), 2)

		if annotate_video:
			writer.writeFrame(frame)
		
		if display_output_on_screen:
			frame = cv2.resize(frame, display_dim)
			cv2.imshow('current frame', frame)
			cv2.waitKey(1)

		print("Frame Number: " +  str(i) + ', Total Detected Tags: ' + str(detected))

		if i == start_end_frames[1]:
			break

		i = i + 1
		
	end_time = datetime.now()
	delta = end_time - start_time
	print('\nExecution time: ' + str(round(delta.total_seconds() * 1000)) + 'ms')
	print('FPS: ' + str(round(float(start_end_frames[1] - start_end_frames[0]) / float(delta.total_seconds()), 2)) + '\n')

	print('Detected total of ' + str(detected) + ' tags.')
	results_df = results_df.astype({'Frame': int,'Tag': int, 'cX': np.float64,'cY': np.float64})
	results_df.to_csv(csv_output_path, index = False)

	# Do a bit of cleanup
	if annotate_video:
		writer.close()
	cv2.destroyAllWindows()
	video_getter.stop()


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


def sigmoid(input_value: float, center: float, slope: float):
	'''
	sigmoid = 1 / (1 + exp[-slope * {input - centur}])
	'''
	return 1 / (1. + np.exp(-slope * (input_value - center)))


def rolling_hungarian_matching(cleaned_aruco_csv_path: str, slp_predictions_path: str, rolling_window_size: int, pairing_threshold: float, sigmoid_inflection: float, sigmoid_slope: float, enhanced_ouput: bool = False) -> list:
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	tag_detection_count = [np.sum(aruco_df.Tag == tag) for tag in tags]

	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	unique_tracks = np.sort(np.unique([j[4] for j in sleap_instances]))

	# Not sure why, but we get a 'track -1'... get rid of it!
	if unique_tracks[0] == -1:
		unique_tracks = unique_tracks[1:-1]
	
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

		# Add the new relevant rows to the rolling window dataframe
		rolling_window_addition = pd.DataFrame(rolling_window_addition)
		rolling_window = rolling_window.append(rolling_window_addition, ignore_index = True)

		if enhanced_ouput:
			print('\n' + '=' * 80)
			print('Current frame: ', current_frame)
			print(rolling_window)

		# Initialize
		this_window_cost_matrix = np.zeros((len(unique_tracks), len(tags)))
		temp_cost_matrix = np.zeros((len(unique_tracks), len(tags)))
		detection_count_matrix = np.zeros((len(unique_tracks), len(tags)))

		last_iter_frame = -1
		for row in rolling_window.itertuples():
			iter_frame = row.Index
			
			if iter_frame != last_iter_frame:
				# We want to add the current frame to the dict of individual frame cost matrices
				# Put instances for current frame in a list
				# Of course, we only do this when we hit a new frame; it's a waste of time to do it for every row because we have many rows per frame!
				iter_frame_instances = []
				iter_sleap_frame = sleap_frames[iter_frame] 
				for iter_frame_idx in range(iter_sleap_frame[3], iter_sleap_frame[4]): # range(instance_id_start, instance_id_end)
					iter_frame_instances.append(sleap_instances[iter_frame_idx])

			# This is the inside of the nested loop; for this particular row, iterate thru the relevant sleap instances.
			for instance in iter_frame_instances:
				start_idx = instance[7]
				# end_idx = instance[8]
				prediction_tuple = sleap_predictions[start_idx] # start_idx corresponds to the tag
				pX = prediction_tuple[0]
				pY = prediction_tuple[1]
				
				if (not np.isnan(pX)) and (not np.isnan(pY)):
					distance = np.sqrt(np.power(float(row.cX) - (pX), 2) + np.power(float(row.cY) - (pY), 2))

					instance_number = int(instance[4])
					if instance_number != -1:
						temp_cost_matrix[tracks_indices[instance_number], tags_indices[int(row.Tag)]] += sigmoid(distance, sigmoid_inflection, sigmoid_slope)
						detection_count_matrix[tracks_indices[instance_number], tags_indices[int(row.Tag)]] += 1
						

			# Keep frame cost matrices in a dict for easy access by frame
			frame_costs_dict[iter_frame] = temp_cost_matrix

			last_iter_frame = iter_frame

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
				detection_count_matrix = np.delete(detection_count_matrix, i, 0)
				trimmed_tracks = np.delete(trimmed_tracks, i)
			else:
				i += 1
		i = 0
		for_hungarian = np.transpose(for_hungarian)
		detection_count_matrix = np.transpose(detection_count_matrix)
		trimmed_tags = np.copy(tags)
		# Eliminate zero-detection columns
		while i < len(for_hungarian):
			if np.sum(for_hungarian[i]) == 0:
				for_hungarian = np.delete(for_hungarian, i, 0)
				detection_count_matrix = np.delete(detection_count_matrix, i, 0)
				trimmed_tags = np.delete(trimmed_tags, i)
			else:
				i += 1


		# Weight entries
		for row_idx in range(len(for_hungarian)):
			row = for_hungarian[row_idx]
			for col_idx in range(len(row)):
				if detection_count_matrix[row_idx, col_idx] > 0:
					for_hungarian[row_idx, col_idx] = float(for_hungarian[row_idx, col_idx] * rolling_window_size) / detection_count_matrix[row_idx, col_idx]

		# Normalize
		for_hungarian = 100. * for_hungarian / np.max(for_hungarian.flatten())

		for row_idx in range(len(for_hungarian)):
			row = for_hungarian[row_idx]
			for col_idx in range(len(row)):
				if detection_count_matrix[row_idx, col_idx] == 0:
					for_hungarian[row_idx, col_idx] = 999.
		

		print('\n\nPrevious cost matrix (processed, rounded to nearest integer, transposed for display purposes):')
		display_matrix = np.transpose(np.vstack([trimmed_tags, np.transpose(for_hungarian)]))
		print(tabulate(display_matrix.astype(int), tablefmt = 'pretty', headers = trimmed_tracks))
		detection_count_matrix = np.transpose(np.vstack([trimmed_tags, np.transpose(detection_count_matrix)]))
		print(tabulate(detection_count_matrix.astype(int), tablefmt = 'pretty', headers = trimmed_tracks))
		hungarian_result = m.compute(np.copy(for_hungarian))
		print('Current frame:               ', current_frame)
		print('Chosen pairs:                ', hungarian_result)
		hungarian_pairs = []
		for tag, track in hungarian_result:
			hungarian_pairs.append((trimmed_tags[tag], trimmed_tracks[track]))
		print('Inferred tag - track pairs:  ', hungarian_pairs)

		# Dictionary of the indices of tags
		trimmed_tags_indices = {}
		for idx in range(len(trimmed_tags)):
			trimmed_tags_indices[trimmed_tags[idx]] = idx

		# Dictionary of the indices of tracks
		trimmed_tracks_indices = {}
		for idx in range(len(trimmed_tracks)):
			trimmed_tracks_indices[trimmed_tracks[idx]] = idx

		# Assignment threshold
		assigned = []
		for tag, track in hungarian_pairs:
			if for_hungarian[trimmed_tags_indices[tag], trimmed_tracks_indices[track]] < pairing_threshold:
				assigned.append(tag)
				tag_tracks_2d_array[tags_indices[tag], current_frame] = track
		print('Tags with new assignments:   ', assigned)

		idx = 0

		if enhanced_ouput:
			 input("Press Enter to continue...")
	
	# Inherit tracks
	for current_frame in range(last_frame):
		for idx in range(len(tag_tracks_2d_array[:, current_frame])):
			if tag_tracks_2d_array[idx, current_frame] == 0 and current_frame > 0:
				tag_tracks_2d_array[idx, current_frame] = tag_tracks_2d_array[idx, current_frame - 1]

	return tag_tracks_2d_array

def import_sleap_instances_to_df(slp_predictions_path: str, fps: float) -> pd.DataFrame:
	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	bee_movements_list = [] # List of tuples to append df rows later; appending to a list is (presumably) faster

	for frame in sleap_frames:
		frame_number = int(frame[0])

		inst_start_idx = int(frame[3])
		inst_end_idx = int(frame[4])
		for instance_idx in range(inst_start_idx, inst_end_idx):
			current_instance = sleap_instances[instance_idx]
			track_number = int(current_instance[4])
			score = float(current_instance[6])

			pred_start_idx = int(current_instance[7])
			pred_end_idx = int(current_instance[8])
			
			# Check if the tag is available otherwise, ignore this point
			tag_prediction = sleap_predictions[pred_start_idx]
			tagX = float(tag_prediction[0])
			tagY = float(tag_prediction[1])

			head_prediction = sleap_predictions[pred_start_idx + 3]
			headX = float(head_prediction[0])
			headY = float(head_prediction[1])

			abdomen_prediction = sleap_predictions[pred_start_idx + 2]
			abdomenX = float(abdomen_prediction[0])
			abdomenY = float(abdomen_prediction[1])

			headAbdomenAngle = np.arctan2(abdomenY - headY, abdomenX - headX)

			if ((not np.isnan(tagX) and not np.isnan(tagY)) and not np.isnan(headAbdomenAngle)) and not np.isnan(score):
				# For now the speeds are left empty; we'll fill them in later

				bee_movements_list.append((frame_number, track_number, score, tagX, tagY, headAbdomenAngle, 0., 0.))

	bee_movements_df = pd.DataFrame(bee_movements_list, columns = ['Frame', 'Track', 'Score', 'tagX', 'tagY', 'headAbdomenAngle', 'tagSpeed', 'headAbdomenRotVel'])
	print(bee_movements_df)

	tracks = np.sort(pd.unique(bee_movements_df['Track']))
	print('\n\nTags: ', tracks)
	print('\n\nCalculating speeds ...\n\n')

	for track in tracks:
		current_track_df = bee_movements_df.loc[bee_movements_df['Track'] == track]

		previous_frame = -2
		previous_row = current_track_df.iloc[0]
		for row in current_track_df.itertuples():
			if row.Frame - previous_frame == 1:
				tag_speed = fps * np.sqrt(np.power(row.tagX - previous_row.tagX, 2) + np.power(row.tagY - previous_row.tagY, 2))
				rot_speed = fps * (row.headAbdomenAngle - previous_row.headAbdomenAngle)

				bee_movements_df.iloc[row.Index, 6] = tag_speed
				bee_movements_df.iloc[row.Index, 7] = rot_speed

			previous_frame = row.Frame
			previous_row = row

	print(bee_movements_df)
	return bee_movements_df


def merge_sleap_tracklets(slp_predictions_path: str, output_path: str, bee_movements_df: pd.DataFrame, pairings: list, change_track_penalty: float, sigmoid_inflection: float, sigmoid_slope: float) -> list:
	m = Munkres()
	frame_dfs = []

	# Variables to avoid running cost analysis on first trigger
	first_trigger = True

	frame = 0
	for row in np.transpose(pairings):
		# Initialize variables relevant to this frame that might need to be used later on
		this_frame_tracks = row
		current_frame_df = bee_movements_df.loc[bee_movements_df['Frame'] == frame]
		frame_dfs.append(current_frame_df)

		# We don't care about consecutive frames with consistent pairings;
		# Only run matching when pairings change.
		# Obviously, we can't run tracklet merging on frame zero; there's nothing to start merging from.
		if frame > 0 and not np.array_equal(this_frame_tracks, last_frame_tracks):
			print('\n' + '=' * 80)
			print(f'Frame {frame}:\n')

			# Remove zeros from arrays; these correspond to unpaired tags
			trimmed_lft = np.ma.masked_equal(np.unique(last_frame_tracks), 0)
			trimmed_lft = trimmed_lft.compressed()
			trimmed_tft = np.ma.masked_equal(np.unique(this_frame_tracks), 0)
			trimmed_tft = trimmed_tft.compressed()
			tft_length = len(trimmed_tft)
			lft_length = len(trimmed_lft)

			print(f'({lft_length}, {tft_length})')
			if tft_length == lft_length:
				print('No tracks gained or lost!')
			elif tft_length > lft_length:
				print('New track out of nowhere!')
			else:
				print("We've lost a track :(")
			
			# Carry over any lost tracks
			trimmed_tft = np.append(trimmed_tft, trimmed_lft)
			trimmed_tft = np.unique(trimmed_tft)
			tft_length = len(trimmed_tft)
			lft_length = len(trimmed_lft)

			print('Old tracks:  ', trimmed_lft)
			print('New tracks:  ', trimmed_tft)

			# Iterate through all track combinations.
			# We want to check if the same track number is actually the same track too
			# So we calculate a cost, even for a track to itself.

			# Initialize cost matrix
			cost_matrix = np.zeros((lft_length, tft_length))

			# Initialize dicts for track indices
			lft_indices = {}
			for idx in range(len(trimmed_lft)):
				lft_indices[trimmed_lft[idx]] = idx

			tft_indices = {}
			for idx in range(len(trimmed_tft)):
				tft_indices[trimmed_tft[idx]] = idx

			for lf_track in trimmed_lft:
				# Try to find entry in bee movements dataframe
				try:
					last_frame_df = frame_dfs[frame - 1]
					lf_track_row = last_frame_df.loc[last_frame_df['Track'] == lf_track]
					lf_track_X = float(lf_track_row['tagX'])
					lf_track_Y = float(lf_track_row['tagY'])
					lf_success = True
				except:
					print(f'On frame {frame - 1} could not retrieve data for track {int(lf_track)}')
					lf_success = False

				for tf_track in trimmed_tft:
					try:
						this_frame_df = frame_dfs[frame]
						tf_track_row = this_frame_df.loc[this_frame_df['Track'] == tf_track]
						tf_track_X = float(tf_track_row['tagX'])
						tf_track_Y = float(tf_track_row['tagY'])
						tf_success = True
					except:
						print(f'On frame {frame} could not retrieve data for track {int(lf_track)}')
						tf_success = False

					# If we could find data for both this frame AND last frame for this track transition combination
					if lf_success and tf_success:

						# Distance cost calculation
						distance = np.sqrt(np.power(tf_track_X - lf_track_X, 2) + np.power(tf_track_Y - lf_track_Y, 2))

						# We use the previously generated dicts to help index on the cost matrix
						cost_matrix[lft_indices[lf_track], tft_indices[tf_track]] += (100.) * sigmoid(distance, sigmoid_inflection, sigmoid_slope)

					# Penalty for changing tracks
					if tf_track != lf_track:
						cost_matrix[lft_indices[lf_track], tft_indices[tf_track]] += change_track_penalty


			# Hungarian matching
			if not first_trigger:
				print(tabulate(cost_matrix.astype('int'), tablefmt = 'pretty'))
				hungarian_result = m.compute(np.copy(cost_matrix))
				print('Chosen pairs:               ', hungarian_result)
				hungarian_pairs = {}
				for lf_track, tf_track in hungarian_result:
					hungarian_pairs[int(trimmed_lft[lf_track])] = int(trimmed_tft[tf_track])
				print('Resultant track transitions:', hungarian_pairs)

				# Change necessary pairings
				frame -= 1
				print('Previous assignments:       ', str(pairings[:, frame]))
				for idx in range(len(pairings[:, frame])):
					if pairings[idx, frame] != 0:
						pairings[idx, frame] = hungarian_pairs[int(pairings[idx, frame])]
				print('Modified assignments:       ', str(pairings[:, frame]))
				frame += 1

			else:
				first_trigger = False

		# Manage history-dependent variables
		last_frame_tracks = np.copy(this_frame_tracks)
		frame += 1

	np.savetxt(output_path, pairings, delimiter = ",")

	return pairings


def hungarian_annotate_video_sleap_aruco_pairings(video_path: str, video_output_path: str, cleaned_aruco_csv_path: str, slp_predictions_path: str,\
 pairings: np.ndarray, tl_coords: tuple[int, int], square_dim: int, end_frame: int, display_output_on_screen: bool = False) -> None:
	'''
	annotate video with pairings of aruco tags and sleap tracks
	output is .avi
	'''
	aruco_df = awpd.load_into_pd_dataframe(cleaned_aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))

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
	success = True
	frame = 0
	while success:
		print('\n' + '=' * 80)
		print(f'Frame {frame}')
		video_data.set(1, frame)
		if square_dim > 0:
			image = image[tl_coords[0]:tl_coords[0] + square_dim, tl_coords[1]:tl_coords[1] + square_dim]
		try:
			success, image = video_data.read()
		except:
			break

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
					print(prediction_tuple)

			try:
				prediction_tuple = sleap_predictions[prediction_start_idx] # start_idx corresponds to the tag
				pX = int(round(prediction_tuple[0]))
				pY = int(round(prediction_tuple[1]))
				image = cv2.circle(image, (pX, pY), 75, (255, 0, 0), 2)
				current_track = nth_inst_tuple[4]
				current_tag = '?'
				for tag_track_idx in range(len(pairings[:, frame])):
					if pairings[tag_track_idx, frame] == current_track:
						current_tag = tags[tag_track_idx]
				cv2.putText(image, str(current_tag), (pX, pY - 75), font, 4, (255, 0, 0), 2)
				cv2.putText(image, str(current_track), (pX, pY + 100), font, 2, (255, 0, 0), 2)
			except:
				print(f'Failure on frame {frame}')

		# Persistent ArUco location plot
		if frame in aruco_df.index:
			for row in aruco_df[aruco_df.index == frame].itertuples():
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
				cv2.putText(image, str(tags[k]), (last_seen[0, k], last_seen[1, k]), font, 2, (0, 0, 255), 2)


		writer.writeFrame(image)
		
		if display_output_on_screen:
			image = cv2.resize(image, (800,800))
			cv2.imshow('', image)
			cv2.waitKey(1)

		current_frame_idx = next_frame_idx
		frame += 1

		if frame == end_frame:
			break

	# Do a bit of cleanup
	video_data.release()
	writer.close()
	cv2.destroyAllWindows()


class VideoGet:
	"""
	Class that continuously gets frames from a VideoCapture object
	with a dedicated thread.
	"""

	def __init__(self, video_path: str, start_frame: int):
		self.stream = cv2.VideoCapture(video_path)
		self.stream.set(1, start_frame)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False

	def start(self):    
		Thread(target=self.get, args=()).start()
		return self

	def get(self):
		while not self.stopped:
			if not self.grabbed:
				self.stop()
			else:
				(self.grabbed, self.frame) = self.stream.read()

	def stop(self):
		self.stopped = True




if __name__ == '__main__':

	# SLEAP data is necessary before running this pipeline; ArUco data will be generated here.
	# We also need the python file aruco_utils_pd.py in the same folder.
	# Video is also assumed to be cropped, since all recent data has been pre-cropped.
	# Cropping functionality can be replaced by changing the 'dimension' value to a positive integer.

	# TODO: allow specification of frames to run pipline on for easier parallelization
	# TODO: write functionality for the pipline to start partway through if data already exists; this speeds up recovery from failures, and can also lay the basis for much more efficient experimentation with parameters!

	# Required arguments when calling this script
	video_path        = str(sys.argv[1])	# The path of the input video; this will be used for generating ArUco tag data and for annotating the result onto video for manual checking
	slp_file_path     = str(sys.argv[2])	# The path of the relevant .slp inference results.
	files_folder_path = str(sys.argv[3])	# The folder which is to contain the plethora of files that this script will spit out.  Please make this ahead of time!
	name_stem         = str(sys.argv[4])	# A basic name stem that will be conjugated to generate the various filenames.
	annotate_video    = bool(sys.argv[5])	# Should we annotate the video with the final pairings
	end_here_frame    = int(sys.argv[6])	# Last frame that should be processed before moving to next step

	# Here's an example of how one might run this:
	# python ./matching_pipline.py ../sleap_videos/20210715_run001_00000000.mp4 ../sleap_videos/20210715_run001_00000000.mp4.predictions.slp ./matching_work 20210715_run001_2min_full_pipline_test True 1000


	# ~~~THE CONTROL PANEL~~~
	# These values are nominally set to what I (dknapp) think to be the best; they are definitely worth messing with though.
	ArUco_tag_ignore_threshold = 30.        # An ArUco tag must be detected on at least (ArUco_tag_ignore_threshold)% of the video frames to be considered
											# Even with high-res video, a few spurious readings sneak through.  This crude data cleaning step sweeps away these rare spurious readings.
											# Setting this too high risks deleting good tags, meaning that bee will be omitted from matching
											# Setting this too low risks a bunch of garbage data cluttering things, slowing things down, and creating more opportunities for the matching to fail in weird ways.

	rolling_Hungarian_window_length = 3     # Length of the rolling window over which tag-track matching costs will be summed for Hungarian matching.
											# Low values are better suited to denser ArUco data; they allow for quicker reactions to changes in tracks
											# Higher values will make for more stubborn matching that reacts sluggishly, but is more accurate than smaller windows sparser ArUco data.

	cost_sigmoid_inflection_distance = 100. # The cost associated with pairing a tag and a track is calculated based on distance between the two when they are simultaneously detected in a frame.
											# This is then passed into a sigmoid function.
											# This parameter specifies the inflection point of the sigmoid cost weighting in pixels distance.
											# Tag-track pairs that are within the pixel distance specified by this parameter will tend to have much lower costs and vice versa.
											# Think of this as a sort of threshold; set it to a value that you expect most good ArUco tag and SLEAP track pairings to be within.

	cost_sigmoid_inflection_slope = 1.      # See the comments on cost_sigmoid_inflection_distance.
											# This parameter specifies how quickly the sigmoid trasitions around the inflection distance.
											# Higher values result in less discrimination of actual distances and focus the matching more on just finding a tag within the threshold
											# Lower values result in more subtle discrimination of distances, but lack the forceful brutality of a hard cut-off
											# This is a parameter that I've not experimented with much, so it's probably worth playing with.

	maximum_assignment_cost_threshold = 10. # Threshold for the maximum cost of a tag-track combination to be actually outputted as a pairing
											# The Hungarian algorithm, given enough tracks, will generate a pairing for EVERY tag to a track.
											# We don't actually want this, because some tags will just have no useful data within the given window.
											# The result is that the Hungarian algorithm is forced to choose amognst only high-cost candidate assignments for the tag.
											# This causes pretty much random track assignment, which is obviously bad on its face, but is even worse at closer inspection.
											# If a tag has no good data within a window, the random assignment could ruin the pairing that would otherwise be inherited from previous frames.
											# The weighting system in Hungarian matching is normalized, and as a general rule of thumb generates a cost between 0 and 100.
											# Most good pairings will have a cost of ~0, so the threshold needn't be set very high at all for matching to go through.
											# If random pairings seem to be being made, lowering this threshold may help.
											# If obvious pairings are not being made, it's worth trying to raise this threshold.

	tracklet_merging_change_penalty = 30.	# In the tracklet merging step, the penalty for changing tracks.
											# As in the first Hungarian matching step, the costs associated with distances are normalized to a 0 to 100 scale.
											# Again, good merges should have a cost of ~0, and bad ones should be in the 90s.
											# However, sometimes the sleap tracks at the borders have dubious data, so we can end up with multiple low-cost track merge possibilities.
											# Erring on the side of caution, this penalty value helps bias the algorithm in favor of staying on the same track (if possible)
											# 30 (default value) in a 0~100 scale is very large, so changes will only happen in cases where the necessity is abundantly clear.
											# Raise this value to make the tracklet merging step more conservative
											# Lower this value to more aggressively tracklet merge to try to preserve natural, smooth paths

	cost_sigmoid_inflection_dist_mt = 100.  # Same concept as cost_sigmoid_inflection_distance, just for tracklet end/start points
											# There is probably a lot of interesting subtelty that goes into setting different values for these, but I haven't really thought about it too much or tried it at all for that matter.

	cost_sigmoid_inflection_slope_mt = 1.   # Same concept as cost_sigmoid_inflection_slope
											# Ditto 2nd line of comments on cost_sigmoid_inflection_dist_mt


	# Redirect mountains of console output to a .txt file
	# TODO: documentation on strategies to sift through the console output.
	with open(files_folder_path + '/' + name_stem + '_console_output.txt', 'w') as f:
		with redirect_stdout(f):

			# Start by producing ArUco labelling
			# This function doesn't return anything, but spits out an output file.
			# We won't use the video annotation functionality of the code, but leave the functionality in place just in case.
			# Setting the parameter dimension to -1 disables the cropping functionality; see top comment block
			# TODO: fix ArUco FPS counter
			ArUco_csv_path = files_folder_path + '/' + name_stem + '_ArUco_tags.csv'
			threaded_aruco_annotate_video(video_path, '', ArUco_csv_path, (0, 0), -1, (0, end_here_frame))

			# Very quick cleaning step, removing less-detected tags.
			ArUco_df = awpd.load_into_pd_dataframe(ArUco_csv_path)
			total_frames = np.max(ArUco_df.index)
			ArUco_df = awpd.threshold_remove_bad_tags(ArUco_df, total_frames, ArUco_tag_ignore_threshold)
			awpd.save_df(ArUco_df, ArUco_csv_path)



			# We can move onto ArUco and SLEAP matching with a rolling window Hungarian matching system
			pairings = rolling_hungarian_matching(ArUco_csv_path, slp_file_path, rolling_Hungarian_window_length, maximum_assignment_cost_threshold, cost_sigmoid_inflection_distance, cost_sigmoid_inflection_slope)



			# Now comes the tracklet merging... I'm not convinced this does too much at the moment, but will continue to tinker with it.

			# Grab the video fps... this isn't necessary for what we're doing at the moment, but it doesn't hurt to do something simple like this for our future selves.
			vs = cv2.VideoCapture(video_path)
			fps = vs.get(cv2.CAP_PROP_FPS)
			vs.release()

			# This next part nominally cleans the .slp file data, lightly processes it, and sticks it into a dataframe.
			# We'll also pickle the dataframe because it's nice to have around, although we won't explicitly use the pickle from here on.
			SLEAP_inference_df = import_sleap_instances_to_df(slp_file_path, fps)
			SLEAP_inference_df.to_pickle(files_folder_path + '/' + name_stem + '_inference_df.pkl')

			# Actually get to the tracklet merging
			pairings = merge_sleap_tracklets(slp_file_path, files_folder_path + '/' + name_stem + '_final_pairings.csv', SLEAP_inference_df, pairings, tracklet_merging_change_penalty, cost_sigmoid_inflection_dist_mt, cost_sigmoid_inflection_slope_mt)



			# We're done with actual data processing!  Yay!
			# Now, we're left with the optional process of annotating the video with our pairings.
			if annotate_video:
				hungarian_annotate_video_sleap_aruco_pairings(video_path, files_folder_path + '/' + name_stem + '_annotated.mp4', ArUco_csv_path, slp_file_path, pairings, (0, 0), -1, end_here_frame, display_output_on_screen = False)


			# Generate dataframe coordinates output
			# Frame, Tag, TagX, TagY
			# TODO: Annotate this
			# TODO: Add all body parts to the dataframe
			sleap_file = sleap_reader(slp_file_path)
			sleap_predictions = np.array(sleap_file['pred_points'])
			sleap_instances   = np.array(sleap_file['instances'])
			sleap_frames = np.array(sleap_file['frames'])
			output_data = []
			for frame in range(end_here_frame):
				current_sleap_frame = sleap_frames[frame]
				current_frame_idx = current_sleap_frame[3]
				next_frame_idx = current_sleap_frame[4]
				for idx in range(current_frame_idx, next_frame_idx):
					nth_inst_tuple = sleap_instances[idx]
					prediction_start_idx = nth_inst_tuple[7]
					prediction_end_idx = nth_inst_tuple[8]
					prediction_coords = [[],[]] # x, y
					for pred_idx in range(prediction_start_idx, prediction_end_idx):
						try:
							prediction_tuple = sleap_predictions[prediction_start_idx] # start_idx corresponds to the tag
							pX = float(round(prediction_tuple[0]))
							pY = float(round(prediction_tuple[1]))
							current_track = nth_inst_tuple[4]
							current_tag = '?'
							for tag_track_idx in range(len(pairings[:, frame])):
								if pairings[tag_track_idx, frame] == current_track:
									current_tag = tags[tag_track_idx]
							if current_tag != '?':
								output_data.append((frame, int(current_tag), pX, pY))
						except:
							print(f'Failure on frame {frame}')

			output_df = pd.DataFrame(output_data, columns = ['Frame', 'Tag', 'TagX', 'TagY'])
			awpd.save_df(output_df, files_folder_path + '/' + name_stem + '_matched_coordinates_dataframe.csv')

