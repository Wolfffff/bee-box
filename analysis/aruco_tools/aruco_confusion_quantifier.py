# Code for tracking ArUco tags in raw videos
# Input is avi and output is labeled avi and tracks as csv

# Heavily based on: https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import aruco_utils_pd as awpd

def create_confusion_matrix(aruco_csv_path: str, output_csv_path: str, checking_window: int):
	aruco_df = awpd.load_into_pd_dataframe(aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	print(tags)

	index_dict = {}
	idx = 0
	for tag in tags:
		index_dict[tag] = idx
		idx += 1

	confusion_matrix = np.zeros((len(tags), len(tags)))

	for frame in range(np.min(aruco_df.index), np.max(aruco_df.index)):
		window_iter = aruco_df.loc[frame:frame + checking_window].itertuples()
		print(window_iter)

		for row1 in window_iter:
			for row2 in window_iter:
				distance = float(np.sqrt(np.power(row2.cX - row1.cX, 2) + np.power(row2.cY - row1.cY, 2)))
				if distance != 0 and row1.Tag != row2.Tag:
					confusion_matrix[index_dict[row1.Tag], index_dict[row2.Tag]] += 1. / distance

	np.savetxt(output_csv_path, confusion_matrix, delimiter = ',')
	print(confusion_matrix)

	plt.imshow(confusion_matrix)
	plt.xticks(range(0, len(tags)), [str(tag) for tag in tags])
	plt.yticks(range(0, len(tags)), [str(tag) for tag in tags])
	plt.colorbar()
	plt.grid()
	plt.show()

create_confusion_matrix('d:\\20210706_run000_00000000_aruco_annotated.csv', 'd:\\confusion_matrix.csv', 50)