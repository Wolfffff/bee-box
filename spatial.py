from shapely.affinity import rotate
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import split, unary_union
from scipy.stats import linregress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import cv2

class SpatialRectangle ():
	def __init__ (self, *args, **kwargs):

		# Confirm only two points were given
		if len(args) != 2: raise Exception (f'Unable to assign SpatialRectangle from: {args}')
		
		# Confirm the coordinates are valid
		for coords in args:
			if len(coords) != 2: raise Exception (f'Unable to assign SpatialRectangle coordinates from: {coords}')

		# Assign the bounding box
		(self._x1, self._y1), (self._x2, self._y2) = args

	@classmethod
	def fromCoords (cls, *args, **kwargs):

		# Return the SpatialRectangle
		return cls(*args, **kwargs)

	def include (self, include_in, x_coords = None, y_coords = None):

		# Check if include_in is a DataFrame
		if isinstance(include_in, pd.DataFrame):
			
			# Require the SpatialRectangle be included in the DataFrame
			return include_in[include_in[x_coords].between(self._x1, self._x2) & include_in[y_coords].between(self._y1, self._y2)]

		else: raise Exception (f'Not implemented')

	def exclude (self, exclude_in, x_coords = None, y_coords = None):

		# Check if exclude_in is a DataFrame
		if isinstance(exclude_in, pd.DataFrame):
			
			# Require the SpatialRectangle be included in the DataFrame
			return exclude_in[~(exclude_in[x_coords].between(self._x1, self._x2) & exclude_in[y_coords].between(self._y1, self._y2))]
			
		else: raise Exception (f'Not implemented')

class SpatialTriangle ():
	def __init__ (self, *args, round_precision = 1, **kwargs):

		# Confirm only two points were given
		if len(args) != 3: raise Exception (f'Unable to assign SpatialTriangle from: {args}')
		
		# Confirm the coordinates are valid
		for coords in args:
			if len(coords) != 2: raise Exception (f'Unable to assign SpatialTriangle coordinates from: {coords}')

		# Assign the triangle
		(self._x1, self._y1), (self._x2, self._y2), (self._x3, self._y3) = args

		# Calculate the distance between the coordinates
		self.a = np.linalg.norm(np.array((self._x1, self._y1)) - np.array((self._x2, self._y2)))
		self.b = np.linalg.norm(np.array((self._x1, self._y1)) - np.array((self._x3, self._y3)))
		self.c = np.linalg.norm(np.array((self._x2, self._y2)) - np.array((self._x3, self._y3)))

		# Calculate the degrees of the three angles
		self.A = round(np.degrees(np.arccos((self.b**2 + self.c**2 - self.a**2) / (2 * self.b * self.c))), round_precision)
		self.B = round(np.degrees(np.arccos((self.c**2 + self.a**2 - self.b**2) / (2 * self.c * self.a))), round_precision)
		self.C = round(np.degrees(np.arccos((self.a**2 + self.b**2 - self.c**2) / (2 * self.a * self.b))), round_precision)

	@classmethod
	def fromCoords (cls, *args, **kwargs):

		# Return the SpatialRectangle
		return cls(*args, **kwargs)

	def max (self, cutoff = 115):

		return max([self.C, self.B]) <= cutoff

	def plot (self):

		triangle_shape = Polygon([(self._x1, self._y1), (self._x2, self._y2), (self._x3, self._y3)])
		plt.plot(*triangle_shape.exterior.xy)

class SpatialInteractor ():
	def __init__ (self, focal_coords = (), origin_coords = (), antifocal_coords = (), centroid_coords = (), skeleton_buffer = 1, semicircle_radius = 300, **kwargs):

		# Assign the body coordinates
		self._focal_coords = focal_coords
		self._origin_coords = origin_coords
		self._antifocal_coords = antifocal_coords

		# Assign the centroid coordinates
		self._centroid_coords = centroid_coords

		# Create a LineString to represent the skeleton of the Interactor
		self._skeleton = LineString([self._focal_coords, self._origin_coords, self._antifocal_coords]).buffer(skeleton_buffer)

		# Create variables to store semicircles
		self._focal_semicircle = self._interactionSemicircle(self._origin_coords, self._focal_coords, semicircle_radius = semicircle_radius)
		self._antifocal_semicircle = None

	@property    
	def _interactor (self):

		# Assign a list of the skeleton and focal semicircle
		interactor_list = [self._skeleton, self._focal_semicircle]

		# Check if an antifocal semicircle is assigned
		if self._antifocal_semicircle != None: interactor_list.append(self._antifocal_semicircle)

		# Return the union
		return unary_union(interactor_list)

	@classmethod
	def fromTest (cls, origin_coords, focal_coords, antifocal_coords = [], **kwargs):

		if not antifocal_coords: antifocal_coords = origin_coords

		# Return the SpatialInteractor
		return cls(focal_coords = focal_coords, origin_coords = origin_coords, antifocal_coords = antifocal_coords, skeleton_buffer = 0.01, semicircle_radius = 1, **kwargs)

	@classmethod
	def fromBeeSeries (cls, bee_series, focal_cols = ['headX', 'headY'], origin_cols = ['thoraxX', 'thoraxY'], antifocal_cols = ['abdomenX', 'abdomenY'], centroid_cols = ['cX', 'cY'], **kwargs):

		# Drop the tag from the series index
		try: bee_series.index = bee_series.index.droplevel(1)
		except: raise Exception(f'Unable to droplevel from series index: {bee_series}')

		# Return the SpatialInteractor
		return cls(focal_coords = bee_series[focal_cols].values, origin_coords = bee_series[origin_cols].values, antifocal_coords = bee_series[antifocal_cols].values, centroid_coords = bee_series[centroid_cols].values, **kwargs)

	@classmethod
	def fromBeeSeriesR (cls, bee_series, focal_cols = ['thoraxX', 'thoraxY'], origin_cols = ['headX', 'headY'], antifocal_cols = ['abdomenX', 'abdomenY'], centroid_cols = ['cX', 'cY'], **kwargs):

		# Drop the tag from the series index
		try: bee_series.index = bee_series.index.droplevel(1)
		except: raise Exception(f'Unable to droplevel from series index: {bee_series}')

		# Return the SpatialInteractor
		return cls(focal_coords = bee_series[focal_cols].values, origin_coords = bee_series[origin_cols].values, antifocal_coords = bee_series[antifocal_cols].values, centroid_coords = bee_series[centroid_cols].values, **kwargs)

	def intersects (self, other_interactor):

		#return self._interactor.intersects(other_interactor._interactor)

		if self._interactor.intersects(other_interactor._interactor): line_style = 'b-'
		else: line_style = 'r-'

		self.plot(line_style, linewidth = 1)
		other_interactor.plot(line_style, linewidth = 1)

		plt.savefig('test.png', dpi = 1200)

		# Focal awareness
		print(self.awareOf(other_interactor), other_interactor.awareOf(self))

		# Angular Distance
		print(self._angularDistance(other_interactor))

		sys.exit()

	def adjacent (self, other_interactor, cutoff = 112.5):

		# Create list to store the SpatialTriangles
		adj_triangles = []

		'''
		Create SpatialTriangles between self and other_interactor (OI) for:
		1) Self: Origin, Focal.		OI: Focal
		2) Self: Origin, AntiFocal.	OI: Focal
		3) Self: Origin, Focal.		OI: AntiFocal
		4) Self: Origin, AntiFocal.	OI: AntiFocal
		'''
		#adj_triangles.append(SpatialTriangle.fromCoords(self._origin_coords, self._focal_coords, other_interactor._focal_coords))
		#adj_triangles.append(SpatialTriangle.fromCoords(self._origin_coords, self._antifocal_coords, other_interactor._focal_coords))
		#adj_triangles.append(SpatialTriangle.fromCoords(self._origin_coords, self._focal_coords, other_interactor._antifocal_coords))
		#adj_triangles.append(SpatialTriangle.fromCoords(self._origin_coords, self._antifocal_coords, other_interactor._antifocal_coords))

		'''
		Create SpatialTriangles between self and other_interactor (OI) for:
		1) Self: Focal, AntiFocal.	OI: Focal
		4) Self: Focal, AntiFocal.	OI: AntiFocal
		'''
		adj_triangles.append(SpatialTriangle.fromCoords(self._focal_coords, self._origin_coords, other_interactor._focal_coords))
		adj_triangles.append(SpatialTriangle.fromCoords(self._focal_coords, self._origin_coords, other_interactor._antifocal_coords))

		# Loop the adjacency SpatialTriangles
		for adj_triangle in adj_triangles:

			'''
			Get the max from angles B & C. These angles are selected
			because they are both related to line segment a (i.e. self).
			'''
			if max(adj_triangle.B, adj_triangle.C) <= cutoff: return True

		return False

	def awareOf (self, other_interactor):

		'''
		Return bool if self is "Aware" of the other_interactor (i.e. visible):
		True: If other_interactor is within the focal semicircle of self
		False: If other_interactor is not within the focal semicircle of self
		'''
		return self._focal_semicircle.intersects(other_interactor._interactor)

	def plot (self, *args, **kwargs):

		# Plot the relevant subunits using pyplot
		#plt.plot(*self._skeleton.xy, *args, **kwargs)
		#plt.plot(*self._focal_semicircle.exterior.xy, *args, **kwargs)
		plt.plot(*self._interactor.exterior.xy, *args, **kwargs)
		if self._antifocal_semicircle: plt.plot(*self._antifocal_semicircle.exterior.xy, *args, **kwargs)

	def _nearParallel (self, other_interactor, cutoffs = ((0, 45), (135, 225), (315, 360))):

		pass

		#print(self._angularDistance(other_interactor))

	def _angularDistance (self, other_interactor, measurement = 'degrees'):

		# Compute the angular distance in radians
		radian_distance = abs(self._focalArcTan() - other_interactor._focalArcTan())

		# Return the angular distance in radians if specified
		if measurement in ['radians', 'radian', 'r']: 
			return radian_distance
		
		# Return the angular distance in degrees if specified
		elif measurement in ['degrees', 'degree', 'd']:
			return np.degrees(radian_distance if np.pi/2 else np.pi - radian_distance)
		
		# Return error if an unknown measurement
		else:
			raise Exception (f'Unknown measurement: {measurement}')

	def _focalArcTan (self):

		# Return the arctan2 from the origin coords to the focal coords
		return self._arcTan2(self._origin_coords, self._focal_coords)

	@staticmethod
	def _arcTan2 (xy1_coords, xy2_coords):

		# Return the arctan2 from xy2 coords to the xy1 coords
		return np.arctan2(*(np.array(xy2_coords) - np.array(xy1_coords))[::-1])

	@staticmethod
	def _interactionSemicircle (origin_coords, focal_coords, extend_by = 2, semicircle_radius = 300, angle_degrees = 135):

		# Create an array of the line
		line_array = np.array([origin_coords, focal_coords])

		# Create a line string of the line
		line_string = LineString(line_array)

		# Assign the distance to extend the line
		line_extend_distance = line_string.length * (extend_by * semicircle_radius)

		# Assign the slope and intercept of the line
		slope, intercept, _, _, _ = linregress(line_array.T)

		# Check if the slope is defined
		if not np.isnan(slope): 

			# Assign if the focal x is less than the origin x
			focal_less_than = focal_coords[0] < origin_coords[0]

			# Assign the extended origin
			extended_origin_x_coord = origin_coords[0] - line_extend_distance * (-1 if focal_less_than else 1)
			extended_origin_y_coord = extended_origin_x_coord * slope + intercept

			# Assign the theta of origin to focal
			theta = SpatialInteractor._arcTan2(origin_coords, focal_coords)#np.arctan2(*(np.array(focal_coords) - np.array(origin_coords))[::-1])

			# Assign the x length using theta
			semicircle_x = semicircle_radius * np.cos(theta)

			# Assign the extended focal
			extended_focal_x_coord = focal_coords[0] + (semicircle_x / 2)
			extended_focal_y_coord = extended_focal_x_coord * slope + intercept

		# Check if the line is vertical, i.e. undefined slope
		else: 

			# Assign if the focal y is less than the origin y
			focal_less_than = focal_coords[1] < origin_coords[1]
			
			# Assign the extended origin
			extended_origin_x_coord = origin_coords[0]
			extended_origin_y_coord = origin_coords[1] - (line_extend_distance * (-1 if focal_less_than else 1))

			# Assign the extended focal
			extended_focal_x_coord = focal_coords[0]
			extended_focal_y_coord = focal_coords[1] + ((semicircle_radius / 2) * (-1 if focal_less_than else 1))

		# Assign the extended line
		extended_line_string = LineString([(extended_origin_x_coord, extended_origin_y_coord), focal_coords])

		# Create the interaction circle
		interaction_circle = Point(focal_coords).buffer(semicircle_radius)

		# Assign the border - to split the circle
		left_border = rotate(extended_line_string, -angle_degrees, origin = Point(focal_coords))
		right_border = rotate(extended_line_string, angle_degrees, origin = Point(focal_coords))

		# Create the split line string
		line_splitter = LineString([*left_border.coords, *right_border.coords[::-1]])

		# Split the interaction circle
		split_interaction_circle = split(interaction_circle, line_splitter)

		# Assign the polygon using the extended focal point
		extended_focal_coords = Point((extended_focal_x_coord, extended_focal_y_coord))

		# Select the polygon that contains the extended focal coords
		if split_interaction_circle.geoms[1].contains(extended_focal_coords): interaction_semicircle = split_interaction_circle.geoms[1]
		else: interaction_semicircle = split_interaction_circle.geoms[0]
		
		# Return the semicircle
		return interaction_semicircle

