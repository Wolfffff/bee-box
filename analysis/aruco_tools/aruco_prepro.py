from decord import VideoReader
from decord import cpu
from matplotlib import pyplot as plt
import numpy as np
import concurrent.futures
import time
import cv2
import skvideo.io

def preprocess_specified_frames_wrapper(p):
	return preprocess_specified_frames(*p)

def preprocess_specified_frames(video_path: str, frames_to_annotate: list, tl_coords: tuple[int,int], dimension: int):
	start = time.perf_counter()
	identity_string = f'[{np.min(frames_to_annotate)}, ..., {np.max(frames_to_annotate)}]'
	print(f'{identity_string} Loading video ...')

	vr = VideoReader(video_path, ctx = cpu(0))
	frames_array = vr.get_batch(frames_to_annotate).asnumpy()
	end_loading = time.perf_counter()

	print(f'{identity_string} Cropping ...')
	# Crop as specified
	frames_array = frames_array[:, tl_coords[0]:tl_coords[0] + dimension, tl_coords[1]:tl_coords[1] + dimension, 0]
	end_cropping = time.perf_counter()

	print(f'{identity_string} Pre-Processing ...')

	for i in range(len(frames_array[:, 0, 0])):
		frame = frames_array[i, :, :]

		# Normalize image
		frame = cv2.normalize(frame, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

		# Otsu's threshold
		frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		# Applying Laplacian transformation:
		frame = cv.Laplacian(frame, cv.CV_8UC3)
	
	end_prepro = time.perf_counter()

	print(f'{identity_string} Finished assignment!')
	print(f'{identity_string} Total time: {round(end_prepro - start, 2)} s, loading: {round(100 *  (start - end_loading)/ (start - end_prepro))}%, cropping: {round(100 *  (end_loading - end_cropping)/ (start - end_prepro))}%, pre-processing: {round(100 *  (end_cropping - end_prepro)/ (start - end_prepro))}%')

	return frames_array


def preprocess_video_multithreaded(video_path: str, video_output_path: str, tl_coords: tuple[int,int], dimension: int, frames_to_annotate: list, frames_per_thread: int, threads: int) -> None:
	'''
	Pre-process video, multithreaded.
	ALWAYS RUN THIS ON A MACHINE WITH OODLES OF RAM.  THIS WILL CRASH YOUR LAPTOP.
	For laptops, synchronously run small chunks using preprocess_specified_frames, writing after each small chunk to keep RAM usage in check
	Output is mp4
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
		results = executor.map(preprocess_specified_frames_wrapper, chunks_to_assign)

	end_prepro = time.perf_counter()

	# Writing with scikit video because unfortunately there isn't writing functionality in Decord
	vs = cv2.VideoCapture(video_path)
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


	for result in results:
		for i in range(len(result[:, 0, 0])):
			writer.writeFrame(result[i, :, :])

	print('Finished!')
	print('\n\n')
	print(f'Time elapsed: {round(end_prepro - start_assign, 2)} s')
	print(f'FPS: {round(float(len(frames_to_annotate)) / (end_prepro - start_assign), 2)}')


def plot_video_brightness_histogram(video_path: str):
	'''
	Randomly sample a frame from video and use them to generate a histogram of the brightness distribution: useful for figuring out preprocessing pipline
	'''
	vr = VideoReader(video_path, ctx = cpu(0))
	total_frames = len(vr)
	vr.skip_frames(np.random.randint(low = 0, high = total_frames))
	
	frame = vr.next().asnumpy()

	plt.hist(frame.ravel(), 256, [0, 256]);
	plt.show()



if __name__ == '__main__':
	# If this is called directly
	# plot_video_brightness_histogram('d:\\20210706_run000_00000000.avi')
	# preprocess_video_multithreaded('d:\\20210706_run000_00000000.avi', 'd:\\prepro_test.mp4', (0, 0), 3648, range(2), 1, 2)