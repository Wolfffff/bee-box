# Code for tracking ArUco tags in raw videos
# Input is avi and output is labeled avi and tracks as csv

# Heavily based on: https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

import sys
import cv2
import skvideo.io
import numpy as np
import pandas as pd
import concurrent.futures
import time
from datetime import datetime
from threading import Thread

import aruco_utils_pd as awpd

# define names of a few possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}

def aruco_annotate_video(video_path: str, video_output_path: str, csv_output_path: str, tl_coords: tuple[int,int], dimension: int, annotate_video: bool = False, display_output_on_screen: bool = False, display_dim: tuple[int, int] = (800, 800)) -> None:
	'''
	Run ArUco on specified video frame by frame, output an avi annotated version and a csv with all of the relevant data.
	'''
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
	vs = cv2.VideoCapture(video_path)
	
	if annotate_video:
		fps = vs.get(cv2.CAP_PROP_FPS)

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

				Theta = np.arctan2(topRight[1] - bottomLeft[1], topRight[0] - bottomLeft[0])

				# 'Frame', 'Tag', 'cX','cY', 'Theta'
				results_df.loc[len(results_df)] = [int(i), int(markerID[0]), cX, cY, Theta]

				if annotate_video:
					frame = cv2.circle(frame, (int(round(cX)), int(round(cY))), 100, (255, 0, 0), 2)
					frame = cv2.line(frame, (int(round(cX - 50 * np.cos(Theta))), int(round(cY - 50 * np.sin(Theta)))), (int(round(cX + 150 * np.cos(Theta))), int(round(cY  + 150 * np.sin(Theta)))), (255, 0, 0), 2)
					cv2.putText(frame, str(int(markerID[0])), (int(round(cX)), int(round(cY)) - 100), font, 2, (255, 0, 0), 2)

		i = i + 1

		if annotate_video:
			writer.writeFrame(frame)
		
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
	results_df = results_df.astype({'Frame': int,'Tag': int, 'cX': np.float64,'cY': np.float64})
	results_df.to_csv(csv_output_path, index=False)

	# Do a bit of cleanup
	vs.release()
	writer.close()
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


def threaded_aruco_annotate_video(video_path: str, video_output_path: str, csv_output_path: str, tl_coords: tuple[int,int], dimension: int, start_end_frames: tuple[int, int], annotate_video: bool = False, display_output_on_screen: bool = False, display_dim: tuple[int, int] = (800, 800)) -> None:
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

	frames_to_watch = 2400
	n_bees = 16

	# Top left x and y for  cropping
	tlx = tl_coords[0]
	tly = tl_coords[1]

	# Assuming this is a square!
	dimension;

	i = start_end_frames[0]
	detected = 0

	while not video_getter.stopped:
		frame = video_getter.frame
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
	results_df.to_csv(csv_output_path, index=False)

	# Do a bit of cleanup
	if annotate_video:
		writer.close()
	cv2.destroyAllWindows()
	video_getter.stop()

	# csv_aruco_df = awpd.load_into_pd_dataframe(csv_output_path)
	# tags = awpd.find_tags_fast(csv_aruco_df)
	# aruco_df_by_tag = awpd.sort_for_individual_tags(csv_aruco_df, tags)

	# for tag in tags:
	# 	print('\n' + '=' * 50)
	# 	print(tag)
	# 	print('\n' + '-' * 50)
	# 	print(aruco_df_by_tag[tag])

	# print('\n\nFound the following tags (total of ' + str(len(tags)) + '):')
	# print(tags)



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
	arucoParams.adaptiveThreshWinSizeStep = 3
	arucoParams.adaptiveThreshWinSizeMin = 3
	arucoParams.adaptiveThreshWinSizeMax = 30

	arucoParams.adaptiveThreshConstant = 12

	arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	arucoParams.errorCorrectionRate = 1.

	print(f'{identity_string} Loading video ...')
	frames_array = read_frames(video_path, frames_to_annotate)
	# vr = VideoReader(video_path, ctx = cpu(0))
	# frames_array = vr.get_batch(frames_to_annotate).asnumpy()
	end_loading = time.perf_counter()

	print(f'{identity_string} Cropping ...')
	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])

	# Crop as specified
	frames_array = frames_array[:, tl_coords[0]:tl_coords[0] + dimension, tl_coords[1]:tl_coords[1] + dimension, 0]
	end_cropping = time.perf_counter()

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
	combined_results_df.reset_index(level = 0, inplace=True)
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



if __name__ == "__main__":
	# threaded_aruco_annotate_video("d:\\20210713_run001_00000000_cut.mp4", "d:\\220210713_run001_00000000_cut_aruco_annotated.mp4", "d:\\20210713_run001_00000000_cut_aruco_annotated.csv", (0, 0), 3660, annotate_video = False, display_output_on_screen = False)
	# aruco_read_video_multithreaded("../sleap_videos/20210713_run001_00000000.mp4", "../sleap_videos/20210713_run001_00000000_aruco_annotated.csv", (0, 0), 3660, range(720000), 100, 8)
	# annotate_video_with_aruco_csv_tags("../sleap_videos/20210713_run001_00000000.mp4", "../sleap_videos/20210713_run001_00000000_aruco_annotated.mp4", "../sleap_videos/20210713_run001_00000000_aruco_annotated.csv", (0, 0), 3660, range(720000))
	# threaded_aruco_annotate_video("d:\\20210713_run001_00000000_cut.mp4", "d:\\20210713_run001_00000000_cut_aruco_annotated.mp4", "d:\\20210713_run001_00000000_cut_aruco_annotated.csv", (0, 0), 3660, (0,1300), annotate_video = True, display_output_on_screen = True)
	# threaded_aruco_annotate_video(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), (0, 0), 3660, (int(sys.argv[4]), int(sys.argv[5])))
	#                             input video       output video path    output data csv               start frame       end frame

	threaded_aruco_annotate_video("d:\\20210715_run001_00000000_cut.mp4", "d:\\20210715_run001_00000000_cut_aruco_annotated.mp4", "d:\\20210715_run001_00000000_cut_aruco_annotated.csv", (0, 0), 3660, (0, 1300), annotate_video = True, display_output_on_screen = False)