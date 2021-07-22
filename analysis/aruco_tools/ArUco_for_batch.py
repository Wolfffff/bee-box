import sys
import cv2
import numpy as np
import pandas as pd

from datetime import datetime
import threading
from contextlib import redirect_stdout



def aruco_annotate_video(video_path: str, csv_output_path: str, start_end_frames: tuple[int, int]) -> None:
	# define names of a few possible ArUco tag OpenCV supports
	ARUCO_DICT = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	}

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

	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY', 'Theta'])

	# loop over the frames from the video stream

	start_time = datetime.now()

	i = start_end_frames[0]
	detected = 0

	vs = cv2.VideoCapture(video_path)
	vs.set(1, start_end_frames[0])

	success = True
	while success:
		success, frame = vs.read()
		
		# detect ArUco markers in the input frame
		(corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

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

		print("Frame Number: " + str(i) + ', Total Detected Tags: ' + str(detected))

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
	vs.release()

if __name__ == '__main__':
	video_path        = str(sys.argv[1])
	csv_output_path   = str(sys.argv[2])
	start_frame       = int(sys.argv[3])
	end_frame         = int(sys.argv[4])

	print('Starting...')
	aruco_annotate_video(video_path, csv_output_path, (start_frame, end_frame))