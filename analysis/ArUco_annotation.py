# USAGE
# python detect_aruco_video.py

# import the necessary packages

import cv2
import numpy as np
import pandas as pd
from datetime import datetime

def annotate_raw_ArUco(video_path: str, output_path: str, tl_coords: tuple[int, int], square_dim: int):
	# define names of each possible ArUco tag OpenCV supports
	ARUCO_DICT = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
		"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
		"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
		"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
		"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
		"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
		"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
		"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
		"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
		"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
		"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
		"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
		"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
		"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
		"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
		"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
		"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
	}


	# load the ArUCo dictionary and grab the ArUCo parameters
	print("[INFO] detecting '{}' tags...".format("DICT_5X5_100"))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
	arucoParams = cv2.aruco.DetectorParameters_create()
	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	arucoParams.adaptiveThreshWinSizeStep = 1
	arucoParams.adaptiveThreshWinSizeMin = 3
	arucoParams.adaptiveThreshWinSizeMax = 30

	arucoParams.adaptiveThreshConstant = 12

	arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	arucoParams.errorCorrectionRate = 1.

	print("[INFO] starting video stream...")
	vs = cv2.VideoCapture(video_path)
	fps = vs.get(cv2.CAP_PROP_FPS)

	out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (800,800))

	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])

	# loop over the frames from the video stream

	start_time = datetime.now()

	font = cv2.FONT_HERSHEY_SIMPLEX

	frames_to_watch = 2400
	bee_population = 16

	detected = 0
	i = 0
	while vs.isOpened():
		vs.set(1, i)
		ret, frame = vs.read()
		frame = frame[tl_coords[0] : tl_coords[0] + square_dim, tl_coords[1] : tl_coords[1] + square_dim]
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

		frame = cv2.resize(frame, (800,800))
		out.write(frame)

		print("Frame Number: " +  str(i) + ', Total Detected Tags: ' + str(detected))
		i += 1

	end_time = datetime.now()
	delta = end_time - start_time
	print('\nExecution time: ' + str(round(delta.total_seconds() * 1000)) + 'ms')
	print('FPS: ' + str(round(float(delta.total_seconds()) / float(frames_to_watch), 2)) + '\n')

	print('Detected total of ' + str(detected) + ' tags.')
	print('This is a data density of ' + str(round(float(detected) * 100. / float(bee_population * frames_to_watch), 2)) + '%')

	results_df = results_df.astype({'Frame': int,'Tag': int, 'cX': np.float64,'cY': np.float64})
	results_df.to_csv("results.csv", index=False)

	# do a bit of cleanup
	vs.release()
	out.release()
	cv2.destroyAllWindows()