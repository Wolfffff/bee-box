import os
import cv2
import h5py
import time
import operator
import skvideo.io
from tabulate import tabulate
from munkres import Munkres
import numpy as np
import pandas as pd

import aruco_utils_pd as awpd

def sleap_reader(slp_predictions_path: str) -> h5py.File:
	f = h5py.File(slp_predictions_path, 'r')
	return f

def read_frames(video_path: str, fidxs: list = None, grayscale: bool = True) -> np.array:
	"""Read frames from a video file.
	Args:
		video_path: Path to MP4
		fidxs: List of frame indices or None to read all frames (default: None)
		grayscale: Keep only one channel of the images (default: True)
	Returns:
		Loaded images in array of shape (n_frames, height, width, channels) and dtype uint8.
	"""
	vr = cv2.VideoCapture(video_path)
	if fidxs is None:
		fidxs = np.arange(vr.get(cv2.CAP_PROP_FRAME_COUNT))
	frames = []
	for fidx in fidxs:
		vr.set(cv2.CAP_PROP_POS_FRAMES, fidx)
		img = vr.read()[1]
		if grayscale:
			img = img[:, :, [0]]
		frames.append(img)
	return np.stack(frames, axis=0)

def crop_run_aruco(video_path: str, slp_predictions_path: str, results_df_path: str, start_end_frame: tuple[int, int], minimum_sleap_score: float = 0.1, crop_size: int = 50, half_rolling_window_size: int = 50, enhanced_output: bool = False, display_images_cv2: bool = False) -> list:
	'''
	Args:
		video_path: path to bee video file.
		slp_predictions_path: path to .slp file containing inference results on relevant video file.
		results_df_path: Where to output the results of SLEAP-based cropping ArUco data.  This data is necessary for video annotation, and probably just great to have in general.
		start_end_frame: Tuple in the form (first frame to process, last frame to process)
		minimum_sleap_score: Minimum SLEAP prediction score for the data point to be used to look for an ArUco tag.  A crude way to try to avoid wasting time on bad SLEAP predicitons.
		crop_size: The number of pixels horizontally and vertically around the SLEAP tag prediction point to crop away to run ArUco on.  Smaller values are faster, but risk cutting off tags and ruining perfectly good data.
		half_rolling_window)size: See next section of the docstring.  Used to specify the size of the rolling window for Hungarian matching.
		enhanced_output: If set to true, this function will spit out a ton of console output useful for debugging.  Best to pipe to a .txt file and carefully read through as necessary.
		display_images_cv2: If set to true, displays cropped images.  Useful for judging crop_size.  Only for local running, and never for actual large batches of data processing.
		
	Frames assignment and rolling window:
		The rolling window is centered at the current frame being processed, with half_rolling_window_size frames on each side.
		This means that for a frame to be processed, it needs to have half_rolling_window_size frames on either side within the assigned range.
		For instance, if I assigned start_end_frame = (1000, 2000) with half_rolling_window_size = 50, the output would have pairings only for frames in the range [1051, 1949]

	Overall process:
		1. Load SLEAP data.
		2. Pull out the SLEAP data we care about from the h5 format so that we can iterate over each SLEAP prediction of a tag. > results_df_path
		3. Around each SLEAP tag prediction, crop a small segment of the frame and run ArUco on this.
		4. Take the data from step 3 to create a rolling-window cost matrix and apply Hungarian matching between tags and tracks. > return pairings in 2d array
	'''
	# This function ....
	# Throw an error if the rolling window is too large for the assigned range of frames; self-explanatory
	if start_end_frame[1] - start_end_frame[0] < (2 * half_rolling_window_size) + 1:
		raise ValueError(f'The rolling window is size (2 * {half_rolling_window_size} + 1) = {(2 * half_rolling_window_size) + 1}, larger than the range of assigned frames which is {start_end_frame[1]} - {start_end_frame[0]} = {start_end_frame[1] - start_end_frame[0]}')

	#
	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	# Not sure why, but we get a 'track -1'... get rid of it!
	unique_tracks = np.sort(np.unique([int(j[4]) for j in sleap_instances]))
	if unique_tracks[0] == -1:
		unique_tracks = unique_tracks[1:-1]

	# Now, let's put the relevant SLEAP tracks into the same data structure: a dict containing interpolated coords for each track
	print('[SLEAP Data Restructuring] Started')
	sleap_interpolation_start = time.perf_counter()

	# Iterate from the start to end frame of the range specified to collect the SLEAP predictions into a simple dataframe
	# We start with everything in a list which is cheaper to append to.  Then we instantiate a dataframe using the list.
	sleap_predictions_df = []
	for frame_number in range(start_end_frame[0], start_end_frame[1] + 1):
		# Throw an exception if we can't find proper SLEAP data in the provided .slp file path.
		try:
			iter_sleap_frame = sleap_frames[frame_number]
		except:
			raise ValueError(f'Provided .slp file does not have predictions for frame {frame_number}, which is within the assigned range of {start_end_frame}!') 
		for iter_frame_idx in range(iter_sleap_frame[3], iter_sleap_frame[4]): # range(instance_id_start, instance_id_end)
			current_instance = sleap_instances[iter_frame_idx]
			prediction_index = current_instance[7] # Member 'point_id_start':  H5T_STD_U64LE (uint64)
			track_number = current_instance[4] # Member 'track':  H5T_STD_I32LE (int32)
			prediction = sleap_predictions[prediction_index]
			if prediction[4] >= minimum_sleap_score: # Member 'score':  H5T_IEEE_F64LE (double)
				# if prediction[2] == 1 and prediction[3] == 1: # Member 'visible':  H5T_ENUM, Member 'complete':  H5T_ENUM
				sleap_predictions_df.append((frame_number, track_number, float(prediction[0]), float(prediction[1])))

	# Instantiate the dataframe
	sleap_predictions_df = pd.DataFrame(sleap_predictions_df, columns = ['Frame', 'Track', 'cX', 'cY'])

	sleap_interpolation_end = time.perf_counter()
	print(f'[SLEAP Data Restructuring] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0] + 1) / float(sleap_interpolation_end - sleap_interpolation_start), 2)}')

	if enhanced_output:
		print(sleap_predictions_df)

	print('[SLEAP-cropped ArUco] Started')
	ScA_start = time.perf_counter()

	print('\n[SLEAP-cropped ArUco] Initializing variables\n')
	# We currently use the collection of 50 tags with 4 x 4 = 16 pixels.
	# define names of a few possible ArUco tag OpenCV supports
	ARUCO_DICT = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	}
	tagset_name = "DICT_4X4_50"
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tagset_name])
	arucoParams = cv2.aruco.DetectorParameters_create()
	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

	# ArUco parameters.  These have been adjusted by dyknapp but are worth playing with if ArUco is too slow or not detecting enough tags
	# These thresholding parameters DRAMATICALLY improve detection rate, while DRAMATICALLY hurting performance.  Since super fast processing isn't really necessary here they should be fine as is.
	arucoParams.adaptiveThreshWinSizeStep = 3
	arucoParams.adaptiveThreshWinSizeMin = 3
	arucoParams.adaptiveThreshWinSizeMax = 30
	# If too slow, start by adjusting this one up.  If we want more tags, lower it (diminishing returns)
	arucoParams.adaptiveThreshConstant = 12

	arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	# If false positives are a problem, lower this parameter.
	arucoParams.errorCorrectionRate = 1.


	# Appending results to a list is cheaper than appending them to a dataframe.
	results_array = []

	# Initialize OpenCV videostream
	# This isn't the fastest way to read video, but it's good enough.
	vs = cv2.VideoCapture(video_path)
	vs.set(cv2.CAP_PROP_POS_FRAMES, start_end_frame[0])

	print('\n[SLEAP-cropped ArUco] Iterating through frames and running ArUco on crops\n')
	# This variable is used to keep track of whether we're still processing the same frame or not.
	# We want to know when we move to a new frame so that we can load it.
	previous_frame = start_end_frame[0] - 1
	# This variable counts tags detected; only relevant for enhanced_output = True
	detections = 0
	for row in sleap_predictions_df.itertuples():
		# If we've moved onto processing a new frame
		if previous_frame != row.Frame:
			# vs.set involves going to the nearest keyframe and then traversing the video from there.
			# As such, it's very computationally intensive.
			# We only want to call it when a frame is skipped; otherwise we're only traversing sequentially anyways.
			# Frame skips should only occur in the UNLIKELY scenario that a frame has NO SLEAP data... Maybe a camera glitch or a temporary obstruction.
			if previous_frame != row.Frame - 1:
					vs.set(cv2.CAP_PROP_POS_FRAMES, int(row.Frame))

			# Load the new frame
			success, frame = vs.read()
			if enhanced_output and row.Frame != start_end_frame[0]:
				print(f'Frame {previous_frame}: {detections} tag(s)')
				detections = 0

		# Only bother with this if the frame could be succesfully loaded.
		# Obviously, no point in trying to run ArUco on junk data
		if success:
			cropped_area = frame[np.maximum(int(row.cY) - crop_size, 0) : np.minimum(int(row.cY) + crop_size, frame.shape[0] - 1), np.maximum(int(row.cX) - crop_size, 0) : np.minimum(int(row.cX) + crop_size, frame.shape[1] - 1), 0]

			# Display cropped tags.
			# This is useful for diagnosing whether the parameter crop_size is set to a reasonable value.
			if display_images_cv2:
				cv2.imshow('cropped area', cv2.resize(cropped_area, (500, 500)))
				cv2.waitKey(0)

			# Run ArUco
			(corners, ids, rejected) = cv2.aruco.detectMarkers(cropped_area, arucoDict, parameters = arucoParams)

			# If we detected any tags
			if len(corners) > 0:
				# Add the number of detected tags to the detections count
				detections += len(corners)

				# Iterate through detected tags and append results to a results list
				# As before in the SLEAP preprocessing step, appending to a list is cheaper so we do so and then instantiate a dataframe.
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

					# Calculate tag rotation.
					# This doesn't really work all that well... doesn't hurt to have though.
					Theta = np.arctan2(topRight[1] - bottomLeft[1], topRight[0] - bottomLeft[0])

					# 'Frame', 'Track', 'Tag', 'cX','cY', 'Theta'
					results_array.append((row.Frame, row.Track, int(markerID[0]), row.cX, row.cY, Theta))

		previous_frame = row.Frame

	if enhanced_output:
		print(f'Frame {previous_frame}: {detections} tag(s)')

	# Instantiate a results dataframe and save it!
	print(f'Writing results of SLEAP-cropped ArUco as csv to {os.path.abspath(results_df_path)}')
	results_df = pd.DataFrame(results_array, columns = ['Frame', 'Track', 'Tag', 'cX','cY', 'Theta'])
	results_df = results_df.astype({'Frame': int, 'Track': int, 'Tag': int, 'cX': np.float64,'cY': np.float64, 'Theta': np.float64})
	results_df.to_csv(results_df_path, index = False)

	ScA_end = time.perf_counter()
	print(f'[SLEAP-cropped ArUco] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0] + 1) / float(ScA_end - ScA_start), 2)}')

	if enhanced_output:
		print(f'\n\nResults dataframe:\n{results_df}\n\n')

	print('[Rolling Window Tag-Track Association] Started')
	RWTTA_start = time.perf_counter()

	print('\n[Rolling Window Tag-Track Association] Initializing variables\n')
	# Munkres is the package for the Hungarian algorithm.
	m = Munkres()

	# This dict stores the cost matrices for individual frames.  The key is the frame number.
	# When we want the cost matrix for a window, we just sum the matrices from the constituent frames.
	frame_cost_matrices_dict = {}

	# Find unique tags and tracks within the data we've collected so far.
	tags = np.sort(pd.unique(results_df['Tag']))
	tracks = np.sort(pd.unique(results_df['Track']))

	# These dicts are useful for indexing in cost matrices (and some others).  The keys are the tag/track numbers, and they return the index for the tag or track number.
	tag_indices = {}
	for idx in range(len(tags)):
		tag_indices[int(tags[idx])] = idx

	track_indices = {}
	for idx in range(len(tracks)):
		track_indices[int(tracks[idx])] = idx

	# Array to store results:
	# Initialize an array full of zeros
	tag_tracks_2d_array = np.zeros((1 + len(tags), start_end_frame[1] - half_rolling_window_size - (start_end_frame[0] + half_rolling_window_size)))
	# Add a column of labels to show which tags are associated to which rows.  This isn't necessary at all on the programming side, but it GREATLY enhances readability for humans trying to debug.
	tag_tracks_2d_array[:, 0] = np.concatenate(([0], tags))
	# Column headers denoting frame numbers.  Same deal as above, although the program does use this, mostly out of convenience.  If it's there for human convenience, we might as well use it when convenient for the program too!
	tag_tracks_2d_array[0, :] = np.concatenate(([start_end_frame[0] - 1], np.arange(start_end_frame[0] + half_rolling_window_size + 1, start_end_frame[1] - half_rolling_window_size)))

	if enhanced_output:
		print('\n')
		print('Detected tags:   ', tags)
		print('Detected tracks: ', tracks)
		print('\n')

	print('\n[Rolling Window Tag-Track Association] Filling initial window\n')
	# Go ahead and fill data for the first window
	# This lets us move forward in the rolling window just by computing the next frame entering the window each time: fast!
	for frame in range(start_end_frame[0], start_end_frame[0] + (2 * half_rolling_window_size) + 1):
		new_frame_df = results_df.loc[results_df['Frame'] == int(frame)]
		
		frame_cost_matrices_dict[frame] = np.zeros((len(tags), len(tracks)))

		for row in new_frame_df.itertuples():
			frame_cost_matrices_dict[frame][tag_indices[int(row.Tag)], track_indices[int(row.Track)]] -= 1

	print('\n[Rolling Window Tag-Track Association] Starting Hungarian assignments with rolling window\n')
	# Start rolling the window forward.
	for center_of_window_frame in range(start_end_frame[0] + half_rolling_window_size + 1, start_end_frame[1] - half_rolling_window_size):
		if enhanced_output:
			print('\n\n' + '=' * 80)
			print(f'Frame (center of window): {center_of_window_frame}\n')

		# Delete data for the frame that's left the window:
		del frame_cost_matrices_dict[center_of_window_frame - half_rolling_window_size - 1]

		# Generate data for frame entering the window:
		# As stated before, we only need to add the results of the very forward-most frame as we roll the window forward.
		frame_cost_matrices_dict[center_of_window_frame + half_rolling_window_size] = np.zeros((len(tags), len(tracks)))
		new_frame_df = results_df.loc[results_df['Frame'] == int(center_of_window_frame) + half_rolling_window_size]
		for row in new_frame_df.itertuples():
			frame_cost_matrices_dict[center_of_window_frame + half_rolling_window_size][tag_indices[int(row.Tag)], track_indices[int(row.Track)]] -= 1

		# Calculate the cost matrix for this window; just by summing over the already-saved individual frame cost matrices.
		# Technically, it's faster to subtract the cost matrix from the one frame leaving the window, then add the cost matrix of the frame entering the window.
		# That compromises readability and copy-paste-ability, and since this really isn't the speed bottleneck anyways, we can let it pass.
		cost_matrix = np.zeros((len(tags), len(tracks)))
		for window_frame in range(center_of_window_frame - half_rolling_window_size, center_of_window_frame + half_rolling_window_size):
			cost_matrix += frame_cost_matrices_dict[window_frame]

		# The Hungarian algorithm is designed for square matrices, and bar coincidence (or perfection on both ArUco and SLEAP sides), there will always be a different number of candidate tracks and tags.
		# The Munkres package has automatic padding, but it still wants the matrix to have more columns then rows when doing so.
		# We transpose the matrix when running the Hungarian algorithm if necessary, to make sure that Munkres is happy.
		# If there are fewer tracks than tags, every track gets a tag, and vice versa.
		if len(tracks) < len(tags):
			cost_matrix = np.transpose(cost_matrix)
			hungarian_result = m.compute(np.copy(cost_matrix))
			hungarian_pairs = []
			for track, tag in hungarian_result:
				hungarian_pairs.append((tags[tag], tracks[track]))
			cost_matrix = np.transpose(cost_matrix)
		else:
			hungarian_result = m.compute(np.copy(cost_matrix))
			hungarian_pairs = []
			for tag, track in hungarian_result:
				if cost_matrix[tag, track] != 0:
					hungarian_pairs.append((tags[tag], tracks[track]))

		# hungarian_pairs is a collection of tuples holding the raw (tag, track) pairing for this particular frame.
		# We want the pairings to be able to change between frames, so we stick them into the tag_tracks_2d_array
		for tag, track in hungarian_pairs:
			# indexing is a bit messy for this array so we make things easier with np.searchsorted.
			# Not optimal, but much more robust and much more readable than stuffing a ton of arithmetic into the index
			tag_tracks_2d_array[1 + tag_indices[tag], np.searchsorted(tag_tracks_2d_array[0, :], center_of_window_frame)] = track

		if enhanced_output:
			display_matrix = np.copy(np.transpose(cost_matrix.astype(int).astype(str)))
			for x in range(display_matrix.shape[0]):
				for y in range(display_matrix.shape[1]):
					if display_matrix[x, y] == '0':
						display_matrix[x, y] = ' '
			print(tabulate(np.transpose(np.vstack([tags.astype(int), display_matrix])), tablefmt = 'pretty', headers = tracks))
			print('\n')
			print('Assigned tag-track pairs: ', hungarian_pairs)

	RWTTA_end = time.perf_counter()
	print(f'\n[Rolling Window Tag-Track Association] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0] + 1) / float(RWTTA_end - RWTTA_start), 2)}')

	# Inherit tracks
	print('\n[Track Inheritance] Running')
	for current_frame in tag_tracks_2d_array[0, 2 : -1]:
		for idx in range(1, len(tags) + 1):
			# indexing is a bit messy for this array so we make things easier with np.searchsorted.
			# Not optimal, but much more robust and much more readable than stuffing a ton of arithmetic into the index
			column_idx = np.searchsorted(tag_tracks_2d_array[0, :], center_of_window_frame)
			if tag_tracks_2d_array[idx, column_idx] == 0 and current_frame > 0:
				tag_tracks_2d_array[idx, column_idx] = tag_tracks_2d_array[idx, column_idx - 1]
	print('\n[Track Inheritance] Finished!')
	print('Done with SLEAP-based cropping ArUco tag-track association (ScAtt)')

	return tag_tracks_2d_array


def annotate_video_sleap_aruco_pairings(video_path: str, video_output_path: str, aruco_csv_path: str, slp_predictions_path: str,\
 pairings: list, frames_to_annotate: list) -> None:
	'''
	annotate video with pairings of aruco tags and sleap tracks
	output is .avi

	Args:
		video_path: path of video to be annotated
		video_output_path: output path for annotated video
		aruco_csv_path: Path with ArUco data
		slp_predictions_path: path of .slp file with relevant inference data for the video
		pairings: the return value of the matching function
		frames_to_annotate: iterable of frames to annotate

	'''
	print('\nStarted video annotation of matching results.')
	aruco_df = awpd.load_into_pd_dataframe(aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])

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
	errors = 0
	for frame in awpd.progressBar(frames_to_annotate):
		video_data.set(1, frame)
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
					errors += 1
			try:
				prediction_tuple = sleap_predictions[prediction_start_idx] # start_idx corresponds to the tag
				pX = int(round(prediction_tuple[0]))
				pY = int(round(prediction_tuple[1]))
				image = cv2.circle(image, (pX, pY), 75, (255, 0, 0), 2)
				current_track = int(nth_inst_tuple[4])
				current_tag = '?'
				for pair in pairings:
					if pair[1] == current_track:
						current_tag = pair[0]
				cv2.putText(image, str(current_tag), (pX, pY - 75), font, 4, (255, 0, 0), 2)
				cv2.putText(image, str(current_track), (pX, pY + 100), font, 2, (255, 0, 0), 2)
			except:
				# print(f'Failure on frame {frame}')
				errors += 1

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
				image = cv2.line(image, (int(round(cX - 50 * np.cos(Theta))), int(round(cY - 50 * np.sin(Theta)))), (int(round(cX + 150 * np.cos(Theta))), int(round(cY  + 150 * np.sin(Theta)))), (255, 0, 0), 2)
				cv2.putText(image, str(tags[k]), (last_seen[0, k], last_seen[1, k]), font, 2, (0, 0, 255), 2)

		writer.writeFrame(image)
		# cv2.imshow('', image)
		# cv2.waitKey(1)

		current_frame_idx = next_frame_idx

	# Do a bit of cleanup
	video_data.release()
	writer.close()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	pairings = crop_run_aruco('d:\\20210715_run001_00000000_cut.mp4', 'd:\\20210725_preds_1200frames.slp', 'd:\\test_results.csv', (0, 1200), minimum_sleap_score = 0.1, crop_size  = 50, half_rolling_window_size = 30, enhanced_output = True, display_images_cv2 = False)
	annotate_video_sleap_aruco_pairings('d:\\20210715_run001_00000000_cut.mp4', 'd:\\20210715_run001_00000000_cut_crop_aruco_test.mp4', 'd:\\test_results.csv', 'd:\\20210725_preds_1200frames.slp', pairings, range(0, 1199))