# Code for tracking ArUco tags in raw videos
# Input is avi and output is labeled avi and tracks as csv

# Heavily based on: https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

import cv2
import skvideo.io
import numpy as np
import pandas as pd
import concurrent.futures
import time
from datetime import datetime
from decord import VideoReader
from decord import cpu, gpu

import aruco_utils_pd as awpd

# define names of a few possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}

def aruco_annotate_video(video_path: str, video_output_path: str, csv_output_path: str, tl_coords: tuple[int,int], dimension: int, display_output_on_screen: bool = False, display_dim: tuple[int, int] = (800, 800)) -> None:
	'''
	Run ArUco on specified video frame by frame, output an avi annotated version and a csv with all of the relevant data.
	'''
	tagset_name = "DICT_4X4_50"
	print("[INFO] detecting '{}' tags...".format(tagset_name))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tagset_name])
	arucoParams = cv2.aruco.DetectorParameters_create()

	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	# arucoParams.adaptiveThreshWinSizeStep = 1
	# arucoParams.adaptiveThreshWinSizeMin = 3
	# arucoParams.adaptiveThreshWinSizeMax = 30

	# arucoParams.adaptiveThreshConstant = 12

	# dknapp constants
	# arucoParams.maxMarkerPerimeterRate = 0.06
	# arucoParams.minMarkerPerimeterRate = 0.03

	#arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	arucoParams.errorCorrectionRate = 1.

	print("[INFO] starting video stream...")
	vs = cv2.VideoCapture(video_path)
	fps = vs.get(cv2.CAP_PROP_FPS)

	out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (dimension, dimension))

	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])

	# loop over the frames from the video stream

	start_time = datetime.now()

	font = cv2.FONT_HERSHEY_SIMPLEX

	frames_to_watch = 2400
	n_bees = 16

	# Top left x and y for  cropping
	tlx = tl_coords[0]
	tly = tl_coords[1]

	# Assuming this is a square!
	dimension;

	i = 0
	detected = 0

	# Iterate over video
	while vs.isOpened():
		ret, frame = vs.read()
		if ret == False:
			break
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

				# 'Frame', 'Tag', 'cX','cY'
				results_df.loc[len(results_df)] = [int(i), int(markerID[0]), cX, cY]

				frame = cv2.circle(frame, (int(round(cX)), int(round(cY))), 50, (255, 0, 0), 2)
				cv2.putText(frame, str(int(markerID[0])), (int(round(cX)), int(round(cY)) - 50), font, 2, (255, 0, 0), 2)

		i = i + 1

		out.write(frame)
		
		if display_output_on_screen:
			frame = cv2.resize(frame, display_dim)
			cv2.imshow('current frame', frame)
			cv2.waitKey(1)

		print("Frame Number: " +  str(i) + ', Total Detected Tags: ' + str(detected))

	end_time = datetime.now()
	delta = end_time - start_time
	print('\nExecution time: ' + str(round(delta.total_seconds() * 1000)) + 'ms')
	print('FPS: ' + str(round(float(delta.total_seconds()) / float(frames_to_watch), 2)) + '\n')

	print('Detected total of ' + str(detected) + ' tags.')
	print('This is a data density of ' + str(round(float(detected) * 100. / float(n_bees * frames_to_watch), 2)) + '%')
	print(f" 10 times 5 is {10 * 5}")
	results_df = results_df.astype({'Frame': int,'Tag': int, 'cX': np.float64,'cY': np.float64})
	results_df.to_csv(csv_output_path, index=False)

	# Do a bit of cleanup
	vs.release()
	out.release()
	cv2.destroyAllWindows()

	csv_aruco_df = awpd.load_into_pd_dataframe(csv_output_path)
	tags = awpd.find_tags_fast(csv_aruco_df)
	aruco_df_by_tag = awpd.sort_for_individual_tags(csv_aruco_df, tags)

	for tag in tags:
		print('\n' + '=' * 50)
		print(tag)
		print('\n' + '-' * 50)
		print(aruco_df_by_tag[tag])

	print('\n\nFound the following tags (total of ' + str(len(tags)) + '):')
	print(tags)


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


def aruco_annotate_specified_frames_wrapper(p):
	return aruco_annotate_specified_frames(*p)

def aruco_annotate_specified_frames(video_path: str, frames_to_annotate: list, tl_coords: tuple[int,int], dimension: int):
	identity_string = f'[{np.min(frames_to_annotate)}, ..., {np.max(frames_to_annotate)}]'
	print(f'{identity_string} Starting ...')
	start = time.perf_counter()
	tagset_name = "DICT_4X4_50"
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tagset_name])
	arucoParams = cv2.aruco.DetectorParameters_create()

	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	# arucoParams.adaptiveThreshWinSizeStep = 1
	# arucoParams.adaptiveThreshWinSizeMin = 3
	# arucoParams.adaptiveThreshWinSizeMax = 30

	# arucoParams.adaptiveThreshConstant = 12

	# dknapp constants
	# arucoParams.maxMarkerPerimeterRate = 0.06
	# arucoParams.minMarkerPerimeterRate = 0.03

	#arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	arucoParams.errorCorrectionRate = 1.

	print(f'{identity_string} Loading video ...')
	# frames_array = read_frames(video_path, frames_to_annotate)
	vr = VideoReader(video_path, ctx = cpu(0))
	with open(video_path, 'rb') as f:
		vr = VideoReader(f, ctx = cpu(0))
	frames_array = np.array(vr.get_batch(frames_to_annotate))
	end_loading = time.perf_counter()

	print(f'{identity_string} Cropping ...')
	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])
	end_cropping = time.perf_counter()

	# Crop as specified
	frames_array = frames_array[:, tl_coords[0]:tl_coords[0] + dimension, tl_coords[1]:tl_coords[1] + dimension, 0]

	print(f'{identity_string} Commencing ArUco tagging ...')
	for i in range(frames_array.shape[0]):
		frame = frames_array[i, :, :]
		# detect ArUco markers in the input frame
		(corners, ids, rejected) = cv2.aruco.detectMarkers(
			frame, arucoDict, parameters=arucoParams)

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

				# 'Frame', 'Tag', 'cX','cY'
				results_df.loc[len(results_df)] = [int(frames_to_annotate[i]), int(markerID[0]), cX, cY]

	end_aruco = time.perf_counter()
	print(f'{identity_string} Finished assignment!')
	print(f'{identity_string} Total time: {round(end_aruco - start, 2)} s, loading: {round(100 *  (start - end_loading)/ (start - end_aruco))}%, cropping: {round(100 *  (end_loading - end_cropping)/ (start - end_aruco))}%, aruco: {round(100 *  (end_cropping - end_aruco)/ (start - end_aruco))}%')

	return results_df


def aruco_read_video_multithreaded(video_path: str, csv_output_path: str, tl_coords: tuple[int,int], dimension: int, frames_to_annotate: list, frames_per_thread: int, threads: int) -> None:
	'''Annotate ArUco tags onto videos with multiprocessing
	'''

	start_assign = time.perf_counter()
	# Prep data to pass
	print('Starting job assignment...')
	assigned_frames = 0
	processes = []
	chunks_to_assign = []
	while assigned_frames < len(frames_to_annotate):
		frames_to_assign = np.minimum(len(frames_to_annotate) - assigned_frames, frames_per_thread)
		frames_to_assign_list = frames_to_annotate[assigned_frames:assigned_frames + frames_to_assign]
		chunks_to_assign.append((video_path, frames_to_assign_list, tl_coords, dimension))
		assigned_frames += frames_to_assign
		print(f'Assigned {assigned_frames} / {len(frames_to_annotate)} frames')

	print('\nStarting parallel processes...')
	with concurrent.futures.ProcessPoolExecutor(max_workers = threads) as executor:
		results = executor.map(aruco_annotate_specified_frames_wrapper, chunks_to_assign)

	end_aruco = time.perf_counter()

	concat_list = []
	for result in results:
		result = result.set_index('Frame')
		concat_list.append(result)
		print(result)
		print('\n\n')

	combined_results_df = pd.concat(concat_list)
	combined_results_df.reset_index(level=0, inplace=True)
	combined_results_df.to_csv(csv_output_path, index=False)
	print('Finished writing results, all concatenated!')
	print(combined_results_df)
	print('\n\n')
	print(f'Time elapsed: {round(end_aruco - start_assign, 2)} s')
	print(f'FPS: {round(float(len(frames_to_annotate)) / (end_aruco - start_assign), 2)}')


def annotate_video_with_aruco_csv_tags(video_path: str, output_video_path: str, csv_input_path: str, tl_coords: tuple[int,int], dimension: int, frames_to_annotate: list):
	print("[INFO] starting video stream...")
	vs = cv2.VideoCapture(video_path)
	fps = vs.get(cv2.CAP_PROP_FPS)
	writer = skvideo.io.FFmpegWriter(output_video_path, outputdict={'-r': str(fps), '-c:v': 'libx264', '-preset': 'veryslow', '-crf' : '0'})

	font = cv2.FONT_HERSHEY_SIMPLEX

	# Load csv
	csv_aruco_df = awpd.load_into_pd_dataframe(csv_input_path)

	print('Annotating...')
	for frame_idx in awpd.progressBar(frames_to_annotate):
		for row in csv_aruco_df[:frame_idx].itertuples():
			vs.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
			ret, frame = vs.read()
			frame = cv2.circle(frame, (int(round(row.cX)), int(round(row.cY))), 50, (255, 0, 0), 2)
			cv2.putText(frame, str(row.Tag), (int(round(row.cX)), int(round(row.cY)) - 50), font, 2, (255, 0, 0), 2)


		frame = np.random.random(size=(5, 480, 680, 3))
		frame = frame.astype(np.uint8)
		writer.writeFrame(frame)

	vs.release()
	writer.close()

if __name__ == "__main__":
	# aruco_annotate_video("d:\\20210629_run000_00000000.avi", "d:\\20210629_run000_00000000_aruco_annotated.avi", "d:\\20210629_run000_00000000_aruco_annotated.csv", (87, 870), 1800)
	aruco_read_video_multithreaded("d:\\20210629_run000_00000000.avi", "d:\\20210629_run000_00000000_aruco_annotated.csv", (87, 870), 1800, range(0, 300), 50, 8)
	# annotate_video_with_aruco_csv_tags("d:\\20210629_run000_00000000.avi", "d:\\20210629_run000_00000000_aruco_annotated.mp4", "d:\\20210629_run000_00000000_aruco_annotated.csv", (87, 870), 1800, range(0, 4))