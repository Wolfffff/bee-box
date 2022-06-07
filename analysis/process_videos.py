#!/usr/bin/env python
import os
import sys
import csv
import argparse

from collections import defaultdict

class BeeVideos (list):
	def __init__ (self, *args, **kwargs):

		# Arguments to store general information
		super(BeeVideos, self).__init__(args[0] if len(args) > 0 else [])

	@property    
	def _all_downloaded (self):
		
		# Check if all videos are downloaded
		return all([_v._all_downloaded for _v in self])

	@property    
	def _all_tracked (self):
		
		# Check if all videos are tracked
		return all([_v._all_tracked for _v in self])

	@property    
	def _all_matched (self):
		
		# Check if all videos are matched
		return all([_v._all_matched for _v in self])

	@classmethod
	def fromDir (cls, input_dir, segment_dict = {}, segment_ext = 'mp4', tracked_ext = 'slp', matched_ext = 'matched_aruco_data_with_track_numbers.csv', matched_suffix = 'matched', **kwargs):

		# Warn user if no segment dict was specified 
		if not segment_dict or len(segment_dict) == 0: print(f'No segment data given. Will process all files within: {input_dir}')

		# Dict to store the video
		video_dict = {}

		# Loop the files within the input dir, process the files
		for root, dirs, files in os.walk(input_dir):
			for input_file in files:
				input_path = os.path.join(root, input_file)
		
				# Assign the input prefix and extension
				segment_name, segment_ext = os.path.splitext(input_file)

				# Assign the segment name
				video_name = segment_name.split('_')[0]

				# Check if a segment dict has been defined
				if segment_dict:

					# Assign the segment number
					try: segment_number = int(segment_name.split('_')[-1])
					except: continue

					# Check if the video is specified (from which the segment was taken)
					if video_name not in segment_dict: continue

					# Check if the current segment is specified
					if segment_number not in segment_dict[video_name]: continue

				# Create the BeeVideo if needed
				if video_name not in video_dict: video_dict[video_name] = BeeVideo(video_name)

				# Add the BeeSegment to the BeeVideo
				video_dict[video_name].append(BeeSegment.fromDir(segment_name, input_path, tracked_ext = tracked_ext, matched_ext = matched_ext, matched_suffix = matched_suffix))

		# Return the BeeVideos object
		return cls(video_dict.values(), **kwargs)

	def trackCmds (self, *args, **kwargs):

		# Check if the videos have been tracked
		if self._all_tracked: return ''

		# String to store the videos track str
		videos_track_str = ''

		# Build the video track str from the segments
		for video in self:
			video_track_str = video._trackCmds(*args, **kwargs)
			if video_track_str: videos_track_str += video_track_str

		# Return the str
		return videos_track_str

	def matchCmds (self, *args, **kwargs):

		# Check if the videos have been tracked
		if self._all_matched: return ''

		# String to store the videos track str
		videos_match_str = ''

		# Build the video track str from the segments
		for video in self:
			video_match_str = video._matchCmds(*args, **kwargs)
			if video_match_str: videos_match_str += video_match_str

		# Return the str
		return videos_match_str

class BeeVideo (list):
	def __init__(self, video_name, *args, **kwargs):

		# Arguments to store general information
		super(BeeVideo, self).__init__(args[0] if len(args) > 0 else [])
		self._video_name = video_name

		# Arguments to store tracking data
		self._models = []

	@property    
	def _all_downloaded (self):
		
		# Check if all segments are downloaded
		return all([_s._downloaded for _s in self])

	@property    
	def _all_tracked (self):
		
		# Check if all segments are tracked
		return all([_s._tracked for _s in self])

	@property    
	def _all_matched (self):
		
		# Check if all segments are matched
		return all([_s._matched for _s in self])

	def __str__(self):
		return(' '.join([str(_s) for _s in self]))

	@classmethod
	def fromDir (cls, video_name, *args, **kwargs):
		return cls(video_name, *args, **kwargs)

	def _trackCmds (self, *args, **kwargs):

		# Check if the video has been tracked
		if self._all_tracked: return ''

		# String to store the video track str
		video_track_str = ''

		# Build the video track str from the segments
		for segment in self:
			segment_track_str = segment._trackCmds(*args, **kwargs)
			if segment_track_str: video_track_str += segment_track_str

		# Return the str
		return video_track_str

	def _matchCmds (self, *args, **kwargs):

		# Check if the video has been tracked
		if self._all_matched: return ''

		# String to store the video track str
		video_match_str = ''

		# Build the video track str from the segments
		for segment in self:
			segment_match_str = segment._matchCmds(*args, **kwargs)
			if segment_match_str: video_match_str += segment_match_str

		# Return the str
		return video_match_str

class BeeSegment ():
	def __init__ (self, segment_name, segment_filename, tracked_ext = '', matched_ext = '', matched_suffix = '', **kwargs):

		# Arguments to store general information
		self._segment_name = segment_name

		# Arguments to drive path
		self._segment_drive_path = ''

		# Arguments to store file extensions
		self._tracked_ext = tracked_ext
		self._matched_ext = matched_ext
		self._matched_suffix = matched_suffix

		# Arguments to the segment filenames
		self._segment_filename = segment_filename
		self._segment_filename_dir = self._segment_filename.rsplit(os.sep, 1)[0]
		self._segment_filename_prefix = os.path.splitext(self._segment_filename)[0]
		self._segment_filename_basename = os.path.basename(self._segment_filename_prefix)

		# Arguments to store the other filenames
		self._tracked_filename = f'{self._segment_filename}.{self._tracked_ext}'
		self._matched_filename = f'{self._segment_filename_prefix}_{self._matched_ext}'
		self._matched_filename_prefix = f'{self._segment_filename_prefix}_{self._matched_suffix}'

		# Arguments to store the status of the segment
		self._downloaded = os.path.isfile(self._segment_filename)
		self._tracked = os.path.isfile(self._tracked_filename)
		self._matched = os.path.isfile(self._matched_filename)

	def __str__(self):
		return(self._segment_name)

	@classmethod
	def fromDir (cls, segment_name, segment_filename, tracked_ext = '', matched_ext = '', matched_suffix = '', **kwargs):
		return cls(segment_name, segment_filename, tracked_ext = tracked_ext, matched_ext = matched_ext, matched_suffix = matched_suffix, **kwargs)

	def _trackCmds (self, model_list):

		# Check if the segment has been tracked
		if self._tracked: return ''

		# Confirm at least one model has been specified
		if len(model_list) == 0: raise Exception(f'No model specified for {self._segment_name}')

		# Create a model substring
		model_substr = ' '.join([f'-m {_m}' for _m in model_list])

		# Return the track command
		return f'sleap-track {self._segment_filename} -o {self._segment_filename}.{self._tracked_ext} {model_substr} --verbosity json --tracking.tracker simple --tracking.similarity centroid --tracker.track_window 2 --tracking.target_instance_count 50 --tracking.post_connect_single_breaks 1\n'

	def _matchCmds (self, end_frame, matching_pipeline = '/tigress/dmr4/bee-box/analysis/aruco_tools/matching_pipeline.py'):

		# Check if the segment has been tracked
		if not self._tracked: 
			print(f'Cannot match untracked segment: {self._segment_filename}')
			return ''

		# Check if the segment has been matched
		if self._matched: return ''

		# Return the match command
		return f'python {matching_pipeline} {self._segment_filename} -o {self._tracked_filename} {self._segment_filename_dir} {self._segment_filename_basename}_{self._matched_suffix} 0 {end_frame} -a -t 4 -w 20 -c 200\n'

def processParser ():
	'''
	Process Pipeline Parser

	Assign the parameters for the process video pipeline

	Parameters
	----------
	sys.argv : list
		Parameters from command lind

	Raises
	------
	IOError
		If the specified files do not exist
	'''

	def confirmFile ():
		'''Custom action to confirm file exists'''
		class customAction(argparse.Action):
			def __call__(self, parser, args, value, option_string=None):
				if not os.path.isfile(value):
					raise IOError('%s not found' % value)
				setattr(args, self.dest, value)
		return customAction

	def confirmDir ():
		'''Custom action to confirm directory exists'''
		class customAction(argparse.Action):
			def __call__(self, parser, args, value, option_string=None):
				if not os.path.isdir(value):
					raise IOError('%s not found' % value)
				setattr(args, self.dest, value)
		return customAction

	def metavarList (var_list):
		'''Create a formmated metavar list for the help output'''
		return '{' + ', '.join(var_list) + '}'

	pipeline_parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	# Input files
	pipeline_parser.add_argument('--process-dir', help = 'Defines the directory to process', type = str, action = confirmDir(), required = True)
	pipeline_parser.add_argument('--segment-file', help = 'Defines the file of video segments to process (as tsv)', type = str, action = confirmFile())

	# Process options
	methods = ('track', 'match')
	pipeline_parser.add_argument('--process-method', metavar = metavarList(methods), help = 'Select the process method', choices = methods, default = 'track', type = str)
	pipeline_parser.add_argument('--track-models', help = 'Tracking models', type = str, nargs = '+')
	pipeline_parser.add_argument('--match-end-frame', help = 'Tracking models', type = int)


	# Output arguments
	pipeline_parser.add_argument('--out-prefix', help = 'Defines the output command file prefix', type = str, default = 'cmds')

	# Return the arguments
	return pipeline_parser.parse_args()

def main():

	# Assign the process args
	process_args = processParser()

	# Create a dict to store the video segments to save
	segment_dict = {}

	# Check if a video segment file was given
	if process_args.segment_file:

		# Populate the segment dict from the segment file
		with open(process_args.segment_file) as segment_file:
			segment_reader = csv.reader(segment_file, delimiter = '\t')
			for video, segments in segment_reader:
				segment_dict[video] = [int(_s.strip()) for _s in segments.split(',')]

	# Store the process information in a BeeVideos
	bee_videos = BeeVideos.fromDir(process_args.process_dir, segment_dict = segment_dict)

	'''
	Check which process to perform:
	1) Track the video using SLEAP
	2) Match the video using python
	3) Report error otherwise
	'''
	if process_args.process_method == 'track':
		
		# Confirm a model has been specified
		if not process_args.track_models: raise Exception(f'Please specify at least one model w/ --track-models')
		
		# Store the commands
		command_str = bee_videos.trackCmds(process_args.track_models)

		# Assign the method suffix
		method_suffix = 'track'

	elif process_args.process_method == 'match':

		# Confirm an end frame has been specified
		if not process_args.match_end_frame: raise Exception(f'Please specify an end frame w/ --match-end-frame')

		# Store the commands
		command_str = bee_videos.matchCmds(process_args.match_end_frame)

		# Assign the method suffix
		method_suffix = 'match'

	else: raise Exception(f'Unknown process specified: {process_args.process_method}')

	# Check if the command str is not populated
	if not command_str: 
		print(f'No {process_args.process_method} processes required')
		return

	# Create the command output file
	command_out_file = open(f'{process_args.out_prefix}.{method_suffix}.txt', 'w')
	command_out_file.write(command_str)
	command_out_file.close()

if __name__== "__main__":
	main()