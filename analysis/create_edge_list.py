import os
import sys
import csv
import itertools
import argparse

import numpy as np
import pandas as pd

from collections import defaultdict

def edgeListParser ():
	'''
	EdgeList Parser

	Assign the parameters for creating the edgelist file

	Parameters
	----------
	sys.argv : list
		Parameters from command line

	Raises
	------
	IOError
		If the specified files do not exist
	'''

	def parser_confirm_file ():
		'''Custom action to confirm file exists'''
		class customAction(argparse.Action):
			def __call__(self, parser, args, value, option_string=None):
				if not os.path.isfile(value):
					raise IOError('%s not found' % value)
				setattr(args, self.dest, value)
		return customAction

	def metavarList (var_list):
		'''Create a formmated metavar list for the help output'''
		return '{' + ', '.join(var_list) + '}'

	edge_list_parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	# Input/Output argument
	edge_list_parser.add_argument('--input', help = 'Defines the input filename', type = str, action = parser_confirm_file(), required = True)
	edge_list_parser.add_argument('--output', help = 'Defines the output filename', type = str, default = 'edge_list.out')
	type_list = ('tsv', 'csv', 'ssv')
	edge_list_parser.add_argument('--output-type', metavar = metavarList(type_list), help = 'Output file type', choices = type_list, default = 'tsv')

	# Cutoff arguments
	edge_list_parser.add_argument('--euclidean-distance-cutoff', help = 'Defines the euclidean distance cutoff', type = int, default = 300)
	edge_list_parser.add_argument('--sample-size-cutoff', help = 'Defines the sample size cutoff', type = int, default = 250)

	# Return the arguments
	return edge_list_parser.parse_args()

def tagEuclideanDistance (row_list):

	# Remove duplicates
	unique_tags = set()
	unique_row_list = [_r for _r in row_list if _r[0] not in unique_tags and not unique_tags.add(_r[0])]

	# Sort for naming, then calculate euclidean distance
	unique_row_list = sorted(unique_row_list, key=lambda n: n[0])	
	for tag1, tag2 in itertools.combinations(unique_row_list, 2):
		yield (tag1[0], tag2[0]), np.linalg.norm(np.array(tag1[1:])-np.array(tag2[1:]))

def inputToEuclideanDistance (input_file, distance_cutoff, sample_size_cutoff):

	with open(input_file) as csv_file:

		# Create dicts to store the ED data
		tagEDAllCount = defaultdict(int)
		tagEDSigCount = defaultdict(int)

		# Create the current frame args
		current_frame_pos = None
		current_frame_data = []

		# Drop the header, check if first column should be dropped
		if not csv_file.readline().split(',')[0]: col_offset = 1
		else: col_offset = 0

		# Read the file as a CSV, 
		csv_reader = csv.reader(csv_file)
		for csv_row in csv_reader:

			# Assign the row
			input_frame_pos, input_tag, input_x, input_y = int(csv_row[0 + col_offset]), csv_row[1 + col_offset], float(csv_row[2 + col_offset]), float(csv_row[3 + col_offset])
			
			# Check if a new frame
			if not current_frame_pos or current_frame_pos != input_frame_pos:

				# Calculate the euclidean distance for each tag combination, if the data exists
				if current_frame_data and len(current_frame_data) > 1: 
					for tag_pair, tag_ed in tagEuclideanDistance(current_frame_data):
						tagEDAllCount[tag_pair] += 1
						if tag_ed < distance_cutoff: tagEDSigCount[tag_pair] += 1

				# Reassign the current frame
				current_frame_pos = input_frame_pos
				current_frame_data = []

			# Add the current line to the data
			current_frame_data.append([input_tag, input_x, input_y])

		# Calculate the euclidean distance for each tag combination, if the data exists
		if current_frame_data and len(current_frame_data) > 1: 
			for tag_pair, tag_ed in tagEuclideanDistance(current_frame_data):
				tagEDAllCount[tag_pair] += 1
				if tag_ed < distance_cutoff: tagEDSigCount[tag_pair] += 1

		for tag_pair, all_count in tagEDAllCount.items():
			if tag_pair not in tagEDSigCount or all_count < sample_size_cutoff: continue
			yield tag_pair[0], tag_pair[1], tagEDSigCount[tag_pair]/all_count

def main():

	# Assign the arguments from the command line
	edge_list_args = edgeListParser()

	# Open the output edge list
	edge_list_output = open(edge_list_args.output, 'w')

	# Calculate the euclidean distance using the input file and cutoff values
	for tag1, tag2, euclidean_distance in inputToEuclideanDistance(edge_list_args.input, edge_list_args.euclidean_distance_cutoff, edge_list_args.sample_size_cutoff):

		# Write the returned (and filtered) euclidean distance to the edge list
		if edge_list_args.output_type == 'tsv': edge_list_output.write(f'{tag1}\t{tag2}\t{euclidean_distance}\n')
		elif edge_list_args.output_type == 'csv': edge_list_output.write(f'{tag1},{tag2},{euclidean_distance}\n')
		elif edge_list_args.output_type == 'ssv': edge_list_output.write(f'{tag1} {tag2} {euclidean_distance}\n')
		else: raise Exception (f'Unable to assign output file type: {edge_list_args.output_type}')

	# Close the output edge list
	edge_list_output.close()

if __name__== "__main__":
	main()