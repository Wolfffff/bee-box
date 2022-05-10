import os
import sys
import logging
import argparse
import itertools
import pandas as pd
import seaborn as sns
import h5py
import palettable
import distinctipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# import utils.trx_utils as trx_utils

from seaborn.distributions import distplot
from matplotlib import colors
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from cv2 import cv2
from matplotlib.animation import FuncAnimation

from spatial import SpatialInteractor, SpatialRectangle


class Tracking:
    def __init__(
        self,
        track_dataframe=pd.DataFrame(),
        coord_pairs=[],
        interaction_coord_pairs=[],
        random_seed=100,
        *args,
        **kwargs,
    ):

        # Confirm the track dataframe is not empty
        if track_dataframe.empty:
            raise Exception(
                f"No data specfied. Tracking requires one or more pd.DataFrame"
            )

        self._track_dataframe = track_dataframe
        self._coord_pairs = coord_pairs
        self._interaction_coord_pairs = interaction_coord_pairs
        self._tracked_tags = list(
            set(self._track_dataframe.columns.get_level_values(1))
        )
        self._interaction_dataframe = pd.DataFrame(
            columns=[
                "Tag_A",
                "Coordinates_A",
                "Tag_B",
                "Coordinates_B",
                "Frame",
                "Distance",
            ]
        )
        self._random_seed = random_seed

        # Confirm there are tags being tracked
        if len(self._tracked_tags) == 0:
            raise Exception("Unable to assign tags")

        # Check the interaction pairs, will need to update interactionRandomDistances if expanded
        interaction_pairs_shape = np.array(self._interaction_coord_pairs).shape
        if len(interaction_pairs_shape) == 1 or interaction_pairs_shape[0] > 1:
            raise Exception("Unable to assign tags interaction pairs")

    @classmethod
    def getDataframeFromFile(cls, filename, *args, **kwargs):
        return cls.fromArucoFile(filename, *args, **kwargs)._track_dataframe

    @classmethod
    def fromListOfArucoFiles(
        cls,
        aruco_files,
        section_length=72000,
        coord_pairs=[
            ["cX", "cY"],
            ["headX", "headY"],
            ["thoraxX", "thoraxY"],
            ["abdomenX", "abdomenY"],
        ],
        interaction_coord_pairs=[["thoraxX", "thoraxY"]],
        non_coord_cols=set(["Frame", "Tag", "Theta"]),
        track_tags=[],
        **kwargs,
    ):
        # experiment_dict["20210909_run000_00000000"]["result_files"][0]

        for k, file in enumerate(aruco_files):

            if not os.path.isfile(file):
                raise Exception(f"File {file} does not exist")

            # Create the track dataframe
            track_df = cls.getDataframeFromFile(
                file,
                coord_pairs=coord_pairs,
                interaction_coord_pairs=interaction_coord_pairs,
                non_coord_cols=non_coord_cols,
                track_tags=track_tags,
                **kwargs,
            )
            track_df = track_df.reindex(
                range(k * section_length, (k + 1) * section_length)
            )

            # Concat the track dataframe
            if k == 0:
                merged_df = track_df
            else:
                merged_df = pd.concat([merged_df, track_df])

        return cls(
            track_dataframe=merged_df,
            coord_pairs=coord_pairs,
            interaction_coord_pairs=interaction_coord_pairs,
            **kwargs,
        )

    @classmethod
    def fromArucoFile(
        cls,
        aruco_file,
        coord_pairs=[
            ["cX", "cY"],
            ["headX", "headY"],
            ["thoraxX", "thoraxY"],
            ["abdomenX", "abdomenY"],
        ],
        interaction_coord_pairs=[["thoraxX", "thoraxY"]],
        non_coord_cols=set(["Frame", "Tag", "Theta"]),
        track_tags=[],
        **kwargs,
    ):

        # Confirm the aruco file is a file
        if not os.path.isfile(aruco_file):
            raise Exception(f"Unable to open {aruco_file}")

        # Open the aruco file as a pd.DataFrame
        aruco_dataframe = pd.read_csv(aruco_file)

        # Check the non-coordinate columns
        if not non_coord_cols.issubset(aruco_dataframe.columns):
            raise Exception("Unable to assign non-coordinate columns")

        # Assign (keeping order) the coordinate columns
        coord_cols = [
            _cs for _cs in aruco_dataframe.columns if _cs not in non_coord_cols
        ]

        # Limit the track coords, if specified
        if coord_pairs:
            matched_coords = []

            # Confirm the coordinate pairs are within the data
            for coord_pair in coord_pairs:
                if not set(coord_pair) < set(coord_cols):
                    raise Exception(f"Coordinate pair not found: {coord_pair}")
                matched_coords.extend(coord_pair)

            # Update the coord cols while keeping order
            coord_cols = [_cs for _cs in coord_cols if _cs in matched_coords]

        # Remove unwated tags
        if track_tags:
            aruco_dataframe = aruco_dataframe[aruco_dataframe.Tag.isin(track_tags)]
        else:
            aruco_dataframe = aruco_dataframe[aruco_dataframe.Tag != -1]

        # Pivot the dataframe
        aruco_dataframe = aruco_dataframe.pivot(
            index="Frame", columns="Tag", values=coord_cols
        )

        # Return the Tracking object
        return cls(
            track_dataframe=aruco_dataframe,
            coord_pairs=coord_pairs,
            interaction_coord_pairs=interaction_coord_pairs,
            **kwargs,
        )

    def interpolate(self, **kwargs):
        self._track_dataframe = self._track_dataframe.interpolate(axis=0, **kwargs)

    def getNumpyArray(self):
        df = self._track_dataframe.copy()
        


    def interactionRandomDistances(
        self,
        include_coords=[],
        exclude_coords=[],
        size_n=1000000,
        round_values=True,
        **kwargs,
    ):

        # Assign the coordinate columns
        coord_pair_x, coord_pair_y = self._interaction_coord_pairs[0]

        # Limit the dataframe to the data of interest
        coords_dataframe = self._track_dataframe.loc[
            :, pd.IndexSlice[[coord_pair_x, coord_pair_y], self._tracked_tags]
        ]

        # Create a list of the coordinate pairs
        coordinate_pairs_list = []
        for tracked_tag in self._tracked_tags:

            # Assign the columns
            col_pair_x, col_pair_y = (
                pd.IndexSlice[coord_pair_x, tracked_tag],
                pd.IndexSlice[coord_pair_y, tracked_tag],
            )

            """
			1) Filter the dataframe to the columns of interest
			2) Round the values, if specified
			3) Restrict coords if given as inclusion coordinates
			4) Remove coords if given as exclusion coordinates
			
			"""
            coords_subset_dataframe = coords_dataframe[
                [col_pair_x, col_pair_y]
            ].dropna()
            if round_values:
                coords_subset_dataframe = coords_subset_dataframe.round()

            # Check if inclusion coordinates were given
            if include_coords:

                # Included the coordinates from the DataFrame
                include_rectangle = SpatialRectangle.fromCoords(*include_coords)
                coords_subset_dataframe = include_rectangle.include(
                    coords_subset_dataframe, x_coords=col_pair_x, y_coords=col_pair_y
                )

            # Check if exclusion coordinates were given
            if exclude_coords:

                # Excluded the coordinates from the DataFrame
                exclude_rectangle = SpatialRectangle.fromCoords(*exclude_coords)
                coords_subset_dataframe = exclude_rectangle.exclude(
                    coords_subset_dataframe, x_coords=col_pair_x, y_coords=col_pair_y
                )

            # Add the values to the list
            coordinate_pairs_list.extend(
                list(
                    zip(
                        coords_subset_dataframe[col_pair_x],
                        coords_subset_dataframe[col_pair_y],
                    )
                )
            )

        # Get the distance between the two random set of coordinate pairs
        distance_list = []
        for random_pair in np.random.randint(
            len(coordinate_pairs_list), size=(size_n, 2)
        ):
            distance_list.append(
                np.linalg.norm(
                    np.array(coordinate_pairs_list[random_pair[0]])
                    - np.array(coordinate_pairs_list[random_pair[1]])
                )
            )

        return pd.Series(distance_list)

    def interactionSampledPairs(
        self,
        include_coords=[],
        exclude_coords=[],
        size_n=1000,
        round_values=True,
        **kwargs,
    ):

        # Assign the coordinate columns
        coord_pair_x, coord_pair_y = self._interaction_coord_pairs[0]

        # Limit the dataframe to the data of interest
        coords_dataframe = self._track_dataframe.loc[
            :, pd.IndexSlice[[coord_pair_x, coord_pair_y], self._tracked_tags]
        ]

        # Create a list of the coordinate pairs
        coordinate_pairs_list = []
        for tracked_tag in self._tracked_tags:

            # Assign the columns
            col_pair_x, col_pair_y = (
                pd.IndexSlice[coord_pair_x, tracked_tag],
                pd.IndexSlice[coord_pair_y, tracked_tag],
            )

            """
			1) Filter the dataframe to the columns of interest
			2) Round the values, if specified
			3) Restrict coords if given as inclusion coordinates
			4) Remove coords if given as exclusion coordinates
			
			"""
            coords_subset_dataframe = coords_dataframe[
                [col_pair_x, col_pair_y]
            ].dropna()
            if round_values:
                coords_subset_dataframe = coords_subset_dataframe.round()

            # Check if inclusion coordinates were given
            if include_coords:

                # Included the coordinates from the DataFrame
                include_rectangle = SpatialRectangle.fromCoords(*include_coords)
                coords_subset_dataframe = include_rectangle.include(
                    coords_subset_dataframe, x_coords=col_pair_x, y_coords=col_pair_y
                )

            # Check if exclusion coordinates were given
            if exclude_coords:

                # Excluded the coordinates from the DataFrame
                exclude_rectangle = SpatialRectangle.fromCoords(*exclude_coords)
                coords_subset_dataframe = exclude_rectangle.exclude(
                    coords_subset_dataframe, x_coords=col_pair_x, y_coords=col_pair_y
                )

            # Add the values to the list
            coordinate_pairs_list.extend(
                list(
                    zip(
                        coords_subset_dataframe[col_pair_x],
                        coords_subset_dataframe[col_pair_y],
                    )
                )
            )

        # Sort the coordinate pairs
        coordinate_pairs_list.sort()

        # Remove duplicates from the pairs
        coordinate_pairs_list = list(
            ukey for ukey, _ in itertools.groupby(coordinate_pairs_list)
        )

        # Create the random number generator with the seed
        rng = np.random.default_rng(self._random_seed)

        # Get the distance between the two random set of coordinate pairs
        distance_list = []

        # Create random samples of length n
        for _ in range(size_n):

            # Create a random sample from the coordinate pairs list
            random_sample = rng.choice(
                len(coordinate_pairs_list), len(self._tracked_tags), replace=False
            )

            # Iterate each combination of the random sample
            for random_pair in itertools.combinations(random_sample, 2):

                # Add the distance to the list
                # distance_list.append(np.linalg.norm(random_pair[0] - random_pair[1]))
                distance_list.append(
                    np.linalg.norm(
                        np.array(coordinate_pairs_list[random_pair[0]])
                        - np.array(coordinate_pairs_list[random_pair[1]])
                    )
                )

        return pd.Series(distance_list)

    def interactionSampledTagPairs(
        self,
        include_coords=[],
        exclude_coords=[],
        size_n=100,
        round_values=True,
        **kwargs,
    ):

        # Assign the coordinate columns
        coord_pair_x, coord_pair_y = self._interaction_coord_pairs[0]

        # Limit the dataframe to the data of interest
        coords_dataframe = self._track_dataframe.loc[
            :, pd.IndexSlice[[coord_pair_x, coord_pair_y], self._tracked_tags]
        ]

        # Create a list of the coordinate pairs
        coordinate_pairs_matrix = []
        for tracked_tag in self._tracked_tags:

            # Assign the columns
            col_pair_x, col_pair_y = (
                pd.IndexSlice[coord_pair_x, tracked_tag],
                pd.IndexSlice[coord_pair_y, tracked_tag],
            )

            """
			1) Filter the dataframe to the columns of interest
			2) Round the values, if specified
			3) Restrict coords if given as inclusion coordinates
			4) Remove coords if given as exclusion coordinates
			
			"""
            coords_subset_dataframe = coords_dataframe[
                [col_pair_x, col_pair_y]
            ].dropna()
            if round_values:
                coords_subset_dataframe = coords_subset_dataframe.round()

            # Check if inclusion coordinates were given
            if include_coords:

                # Included the coordinates from the DataFrame
                include_rectangle = SpatialRectangle.fromCoords(*include_coords)
                coords_subset_dataframe = include_rectangle.include(
                    coords_subset_dataframe, x_coords=col_pair_x, y_coords=col_pair_y
                )

            # Check if exclusion coordinates were given
            if exclude_coords:

                # Excluded the coordinates from the DataFrame
                exclude_rectangle = SpatialRectangle.fromCoords(*exclude_coords)
                coords_subset_dataframe = exclude_rectangle.exclude(
                    coords_subset_dataframe, x_coords=col_pair_x, y_coords=col_pair_y
                )

            # Add the values to the matrix
            coordinate_pairs_matrix.append(
                list(
                    zip(
                        coords_subset_dataframe[col_pair_x],
                        coords_subset_dataframe[col_pair_y],
                    )
                )
            )

        for coordinate_pairs_pos in range(len(self._tracked_tags)):

            # Sort the coordinate pairs
            coordinate_pairs_matrix[coordinate_pairs_pos].sort()

            # Remove duplicates from the pairs
            coordinate_pairs_matrix[coordinate_pairs_pos] = list(
                ukey
                for ukey, _ in itertools.groupby(
                    coordinate_pairs_matrix[coordinate_pairs_pos]
                )
            )

        # Create the random number generator with the seed
        rng = np.random.default_rng(self._random_seed)

        # Get the distance between the two random set of coordinate pairs
        distance_list = []

        # Create random samples of length n
        for _ in range(size_n):

            # Create a random sample from the coordinate pairs list
            random_sample = (
                []
            )  # rng.choice(coordinate_pairs_list, len(self._tracked_tags), replace = False)
            for coordinate_pairs_pos in range(len(self._tracked_tags)):
                random_pos = rng.integers(
                    len(coordinate_pairs_matrix[coordinate_pairs_pos])
                )
                random_sample.append(
                    np.array(coordinate_pairs_matrix[coordinate_pairs_pos][random_pos])
                )

            # Iterate each combination of the random sample
            for random_pair in itertools.combinations(random_sample, 2):

                # Add the distance to the list
                distance_list.append(np.linalg.norm(random_pair[0] - random_pair[1]))
                # distance_list.append(np.linalg.norm(np.array(coordinate_pairs_list[random_pair[0]]) - np.array(coordinate_pairs_list[random_pair[1]])))

        return pd.Series(distance_list)

    def interactionUniformRandom(self, size_n=100000, **kwargs):

        coord_min = self._track_dataframe.min().min()
        coord_max = self._track_dataframe.max().max()

        # Create the random number generator with the seed
        rng = np.random.default_rng(self._random_seed)

        return pd.Series(
            np.linalg.norm(
                rng.integers(coord_min, coord_max, size=(size_n, 2))
                - rng.integers(coord_min, coord_max, size=(size_n, 2)),
                axis=1,
            )
        )

    def interactionDistances(
        self, include_coords=[], exclude_coords=[], round_values=False
    ):

        ## Only create the interaction dataframe if necessary
        # if not self._interaction_dataframe.empty: return

        # Create a list to store the interactions
        interaction_list = []

        # Loop unique tag combinations
        for tagA, tagB in itertools.combinations(self._tracked_tags, 2):

            # Create a list to store the dataframes of each coord pair combination
            tag_pair_interactions = []

            """
			Loop the product of all coordinates tags. For example, we require:
			A) tagA(cX, cY) vs. tagB(headX, headY) &
			   tagA(headX, headY) vs. tagB(cX, cY) - Not possible w/ combinations()
			B) tagA(cX, cY) vs. tagB(cX, cY) - Not possible w/ permutations() or combinations()
			"""
            for tagA_coord_pair, tagB_coord_pair in itertools.product(
                self._interaction_coord_pairs, repeat=2
            ):

                # Assign the x and y coords
                tagA_pair_x, tagA_pair_y = (
                    pd.IndexSlice[tagA_coord_pair[0], tagA],
                    pd.IndexSlice[tagA_coord_pair[1], tagA],
                )
                tagB_pair_x, tagB_pair_y = (
                    pd.IndexSlice[tagB_coord_pair[0], tagB],
                    pd.IndexSlice[tagB_coord_pair[1], tagB],
                )

                # Create the tags dataframe
                tags_dataframe = pd.merge(
                    self._track_dataframe.loc[:, [tagA_pair_x, tagA_pair_y]],
                    self._track_dataframe.loc[:, [tagB_pair_x, tagB_pair_y]],
                    left_index=True,
                    right_index=True,
                ).dropna()

                # Round the values, if specified
                if round_values:
                    tags_dataframe = tags_dataframe.round()

                # Check if inclusion coordinates were given
                if include_coords:

                    # Included the coordinates from the DataFrame
                    include_rectangle = SpatialRectangle.fromCoords(*include_coords)
                    tags_dataframe = include_rectangle.include(
                        tags_dataframe, x_coords=tagA_pair_x, y_coords=tagA_pair_y
                    )
                    tags_dataframe = include_rectangle.include(
                        tags_dataframe, x_coords=tagB_pair_x, y_coords=tagB_pair_y
                    )

                # Check if exclusion coordinates were given
                if exclude_coords:

                    # Excluded the coordinates from the DataFrame
                    exclude_rectangle = SpatialRectangle.fromCoords(*exclude_coords)
                    tags_dataframe = exclude_rectangle.exclude(
                        tags_dataframe, x_coords=tagA_pair_x, y_coords=tagA_pair_y
                    )
                    tags_dataframe = exclude_rectangle.exclude(
                        tags_dataframe, x_coords=tagB_pair_x, y_coords=tagB_pair_y
                    )

                # Create a new dataframe to save the interaction data
                interaction_dataframe = pd.DataFrame(
                    index=tags_dataframe.index,
                    columns=[
                        "Tag_A",
                        "Coordinates_A",
                        "Tag_B",
                        "Coordinates_B",
                        "Frame",
                        "Distance",
                    ],
                )

                # Assign all non-interaction data to the dataframe
                interaction_dataframe["Tag_A"] = tagA
                interaction_dataframe["Tag_B"] = tagB
                interaction_dataframe["Coordinates_A"] = ", ".join(tagA_coord_pair)
                interaction_dataframe["Coordinates_B"] = ", ".join(tagB_coord_pair)
                interaction_dataframe["Frame"] = interaction_dataframe.index

                # Assign the distance
                interaction_dataframe["Distance"] = np.linalg.norm(
                    tags_dataframe.loc[:, [tagA_pair_x, tagA_pair_y]].values
                    - tags_dataframe.loc[:, [tagB_pair_x, tagB_pair_y]].values,
                    axis=1,
                )

                # Add the current combination to the tag pair list
                tag_pair_interactions.append(interaction_dataframe)

            # Create a DataFrame of the tag pair
            tag_pair_dataframe = pd.concat(tag_pair_interactions, ignore_index=True)

            # Sort the interactions and select the best interaction between the two tags per frame
            tag_pair_dataframe = tag_pair_dataframe.sort_values(
                by=["Tag_A", "Tag_B", "Frame", "Distance"]
            )
            tag_pair_dataframe = tag_pair_dataframe.drop_duplicates(
                subset=["Tag_A", "Tag_B", "Frame"]
            )

            # Add the best interactions to the complete interaction list
            interaction_list.append(tag_pair_dataframe)

        # Save the interactions as a DataFrame
        self._interaction_dataframe = pd.concat(interaction_list)

    def reportInteractionEnrichment(
        self,
        file_prefix,
        *args,
        null_model="uniform",
        include_last_step=True,
        bin_enrichment=True,
        **kwargs,
    ):

        # Confirm the required number of arguments
        if len(args) != 3:
            raise Exception("reportInteractionEnrichment requires three arguments")

        # Generate random interaction distances
        if null_model.lower() == "emperical":
            random_distances = self.interactionSampledPairs(**kwargs)
        elif null_model.lower() == "uniform":
            random_distances = self.interactionUniformRandom()
        else:
            raise Exception(f"Selected null model not supported: {null_model}")

        # Other nulls, likely should just be removed
        # random_distances = self.interactionRandomDistances(**kwargs)
        # random_distances = self.interactionSampledTagPairs(**kwargs)

        # Assign the emperical interaction distances per frame
        self.interactionDistances()

        # Create a series of the emperical interaction distances
        emperical_distances = self._interaction_dataframe["Distance"]

        self.disPlot(
            random_distances, file_prefix=file_prefix, file_suffix=f"Null-{null_model}"
        )
        self.disPlot(
            emperical_distances, file_prefix=file_prefix, file_suffix=f"Emperical"
        )

        # Assign the total count of the two datasets
        total_emperical_distance_count = emperical_distances.size
        total_random_distance_count = random_distances.size

        # Assign the range variables
        interaction_start, interaction_end, interaction_step = args

        # Check if the last step is requested
        if include_last_step:
            interaction_end += interaction_step

        # Create a list to store the enrichment data
        enrichment_list = []

        # Loop the distance thresholds
        for interaction_threshold in range(
            interaction_start, interaction_end, interaction_step
        ):

            # Assign the interaction threshold minimum
            interaction_threshold_min = interaction_threshold - interaction_step
            if not bin_enrichment:
                interaction_threshold_min = 0

            # Assign the thresholded counts
            thresholded_emperical_distance_count = emperical_distances[
                (emperical_distances >= interaction_threshold_min)
                & (emperical_distances < interaction_threshold)
            ].size
            thresholded_random_distance_count = random_distances[
                (random_distances >= interaction_threshold_min)
                & (random_distances < interaction_threshold)
            ].size

            # Calculate the Fisher's exact test for the current threshold
            oddsratio, p_value = stats.fisher_exact(
                np.array(
                    [
                        [
                            thresholded_emperical_distance_count,
                            thresholded_random_distance_count,
                        ],
                        [total_emperical_distance_count, total_random_distance_count],
                    ]
                ),
                alternative="greater",
            )
            fold_difference = self.foldDifference(
                (thresholded_emperical_distance_count / total_emperical_distance_count),
                (thresholded_random_distance_count / total_random_distance_count),
            )
            enrichment_list.extend(
                [
                    {
                        "Distance": interaction_threshold_min,
                        "P-Value": p_value,
                        "Fold Difference": fold_difference,
                    }
                ]
            )

        # Create a dataframe from the list
        enrichment_dataframe = pd.DataFrame(enrichment_list)

        self.linePlot(
            enrichment_dataframe,
            x_axis="Distance",
            y_axis="P-Value",
            y2_axis="Fold Difference",
            file_prefix=file_prefix,
            file_suffix=f"Null-{null_model}",
        )

    def report1dVelocity(
        self,
        file_prefix,
        lag=1,
        subplot_max_value=500,
        violin_plot=True,
        report_min_zscore=None,
        report_min_value=None,
    ):

        # Create a list to store the velocity data
        merged_velocity_list = []

        # Loop the dataframe by column
        for col in self._track_dataframe.columns:

            # Calculate the distance between non-NaN values by the lag value
            col_dataframe = self._track_dataframe[[col]].dropna()

            # Create a new dataframe to save the velocity data
            velocity_dataframe = pd.DataFrame(
                index=col_dataframe.index,
                columns=["Tag", "Coordinate", "Start Frame", "End Frame", "Velocity"],
            )

            ## Assign all non-velocity to the dataframe
            velocity_dataframe["Tag"] = col[1]
            velocity_dataframe["Coordinate"] = col[0]
            velocity_dataframe["End Frame"] = velocity_dataframe.index
            velocity_dataframe["Start Frame"] = velocity_dataframe["End Frame"].shift(
                periods=lag
            )

            # Assign the velocity using the Euclidean Distance and the difference between frames
            velocity_dataframe["Velocity"] = abs(
                col_dataframe[col].diff(periods=lag)
            ) / velocity_dataframe["End Frame"].diff(periods=lag)

            # Drop missing data, and change the type of the end frame
            velocity_dataframe = velocity_dataframe.dropna()
            velocity_dataframe["Start Frame"] = velocity_dataframe[
                "Start Frame"
            ].astype(int)

            # Append the merged velocity list
            merged_velocity_list.append(velocity_dataframe)

        # Create the merged velocity DataFrame
        merged_velocity_dataframe = pd.concat(merged_velocity_list)

        # Create a violin plot of the report
        if violin_plot:
            self.violinPlot(
                merged_velocity_dataframe, y_axis="Velocity", file_prefix=file_prefix
            )
            self.violinPlot(
                merged_velocity_dataframe,
                y_axis="Velocity",
                file_prefix=file_prefix,
                max_y_value=subplot_max_value,
            )

        # Filter the report by a min zscore
        if report_min_zscore != None:
            merged_velocity_dataframe = self.filterMinZscore(
                merged_velocity_dataframe,
                column="Velocity",
                min_zscore=report_min_zscore,
            )

        # Filter the report by a min value
        if report_min_value != None:
            merged_velocity_dataframe = self.filterMinValue(
                merged_velocity_dataframe, column="Velocity", min_value=report_min_value
            )

        # Create the report dataframe
        merged_velocity_dataframe.to_csv(f"{file_prefix}.tsv", sep="\t", index=False)

    def report2dVelocity(
        self,
        file_prefix,
        lag=1,
        subplot_max_value=500,
        violin_plot=True,
        report_min_zscore=None,
        report_min_value=None,
    ):
        def colsEuclideanDistance(cols_dataframe):

            # Return the euclidean distance
            return np.linalg.norm(
                cols_dataframe.values - cols_dataframe.shift(periods=lag).values, axis=1
            )

        # Create a list to store the velocity data
        merged_velocity_list = []

        for coord_cols, tag_col in itertools.product(
            self._coord_pairs, self._tracked_tags
        ):

            # Create a datafrane of the columns without missing data
            cols_dataframe = self._track_dataframe.loc[
                :, pd.IndexSlice[coord_cols, tag_col]
            ].dropna()

            # Create a new dataframe to save the velocity data
            velocity_dataframe = pd.DataFrame(
                index=cols_dataframe.index,
                columns=["Tag", "Coordinates", "Start Frame", "End Frame", "Velocity"],
            )

            # Assign all non-velocity data to the dataframe
            velocity_dataframe["Tag"] = tag_col
            velocity_dataframe["Coordinates"] = ", ".join(coord_cols)
            velocity_dataframe["End Frame"] = velocity_dataframe.index
            velocity_dataframe["Start Frame"] = velocity_dataframe["End Frame"].shift(
                periods=lag
            )

            # Assign the velocity using the Euclidean Distance and the difference between frames
            velocity_dataframe["Velocity"] = colsEuclideanDistance(
                cols_dataframe
            ) / velocity_dataframe["End Frame"].diff(periods=lag)

            # Drop missing data, and change the type of the end frame
            velocity_dataframe = velocity_dataframe.dropna()
            velocity_dataframe["Start Frame"] = velocity_dataframe[
                "Start Frame"
            ].astype(int)

            # Append the mrged velocity list
            merged_velocity_list.append(velocity_dataframe)

        # Create the merged velocity DataFrame
        merged_velocity_dataframe = pd.concat(merged_velocity_list)

        # Create a violin plot of the report
        if violin_plot:
            self.violinPlot(
                merged_velocity_dataframe, y_axis="Velocity", file_prefix=file_prefix
            )
            self.violinPlot(
                merged_velocity_dataframe,
                y_axis="Velocity",
                file_prefix=file_prefix,
                max_y_value=subplot_max_value,
            )

        # Filter the report by a min zscore
        if report_min_zscore != None:
            merged_velocity_dataframe = self.filterMinZscore(
                merged_velocity_dataframe,
                column="Velocity",
                min_zscore=report_min_zscore,
            )

        # Filter the report by a min value
        if report_min_value != None:
            merged_velocity_dataframe = self.filterMinValue(
                merged_velocity_dataframe, column="Velocity", min_value=report_min_value
            )

        # Create the report dataframe
        merged_velocity_dataframe.to_csv(f"{file_prefix}.tsv", sep="\t", index=False)

    @staticmethod
    def disPlot(data_list, file_prefix="", file_suffix="", file_type="png"):

        # Create the distance plot
        dis_plot = sns.displot(data=data_list, kde=True)

        # Save the plot to a file
        dis_plot.savefig(f"{file_prefix}_Distance{file_suffix}.{file_type}")

        # Close the plot
        plt.close()

    @staticmethod
    def linePlot(
        dataframe,
        x_axis="",
        y_axis="",
        y2_axis="",
        file_prefix="",
        file_suffix="",
        file_type="png",
    ):

        # Check if X/Y-axis have been defined
        if not x_axis or not y_axis:
            raise Exception("X-axis and/or Y-axis not assigned")

        fig, ax1 = plt.subplots()

        color = "tab:blue"
        ax1.set_xlabel("Distance (pixels)")
        ax1.set_ylabel("P-Value", color=color)
        ax1.step(data=dataframe, x=x_axis, y=y_axis, color=color)
        ax1.fill_between(
            dataframe[x_axis], dataframe[y_axis], step="pre", alpha=0.5, color=color
        )

        ax2 = ax1.twinx()

        color = "tab:red"
        ax2.set_ylabel(
            "Fold Enrichment", color=color
        )  # we already handled the x-label with ax1
        ax2.step(data=dataframe, x=x_axis, y=y2_axis, color=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f"{file_prefix}_EnrichmentPlot{file_suffix}.{file_type}")

    @staticmethod
    def violinPlot(
        dataframe,
        y_axis="",
        file_prefix="",
        file_type="png",
        x_axis="Tag",
        max_y_value=None,
    ):
        def setColors(color_dataframe):

            # Create a dict of the values
            value_dict = color_dataframe.groupby(by=[x_axis])[y_axis].mean().to_dict()

            # Assign color map and normalized colors
            cmap = plt.cm.get_cmap("coolwarm")  # coolwarm_r
            color_norm = colors.Normalize(
                vmin=min(value_dict.values()), vmax=max(value_dict.values())
            )

            # Return the colors
            return {_k: cmap(color_norm(_v)) for _k, _v in value_dict.items()}

        # Adjust the size of the figure to better suit the number of tags
        plt.figure(figsize=(32, 6))

        # Check if Y-axis has been defined
        if not y_axis:
            raise Exception("Y-axis not assigned")

        # Set the Seaborn theme
        sns.set_theme(style="whitegrid")

        # Create a variable to store additional suffix
        plot_suffix = ""

        # Check if the dataframe should not be filtered
        if max_y_value == None:

            # Create and save a violin plot of the distances
            violin_plot = sns.violinplot(
                x=x_axis,
                y=y_axis,
                data=dataframe,
                palette=setColors(dataframe),
                cut=0,
                linewidth=0.05,
            )

        # Check if the dataframe should be filtered
        else:

            # Filter the dataframe using the max y axis
            filtered_dataframe = dataframe[dataframe[y_axis] < max_y_value]

            # Update the suffix to relect the plot has been filtered
            plot_suffix = "subplot."

            # Create and save a violin plot of the filtered distances
            violin_plot = sns.violinplot(
                x=x_axis,
                y=y_axis,
                data=filtered_dataframe,
                palette=setColors(dataframe),
                cut=0,
                linewidth=0.05,
            )

        # Save the plot
        violin_fig = violin_plot.get_figure()
        violin_fig.savefig(f"{file_prefix}_ViolinPlot.{plot_suffix}{file_type}")

        # Close the plot
        plt.close()

    @staticmethod
    def filterMinZscore(dataframe, column="", min_zscore=None):

        # Confirm a column has been assgined
        if not column:
            raise Exception("No column assigned to exclude non-outliers")

        # Confirm a min zscore was given
        if min_zscore == None:
            raise Exception("No min zscore assigned to exclude non-outliers")

        # Exclude non-outliers from the dataframe
        return dataframe[(np.abs(stats.zscore(dataframe[column])) > min_zscore)]

    @staticmethod
    def filterMinValue(dataframe, column="", min_value=None):

        # Confirm a column has been assgined
        if not column:
            raise Exception("No column assigned to exclude non-outliers")

        # Confirm a min value was given
        if min_value == None:
            raise Exception("No min value assigned to exclude non-outliers")

        # Exclude non-outliers from the dataframe
        return dataframe[(np.abs(dataframe[column]) < min_value)]

    @staticmethod
    def foldDifference(*args):

        # Confirm the calculation is possible
        if len(args) != 2:
            raise Exception(f"Unable to calculate fold difference: {args}")

        # Calcuate the fold difference, return nan if impossible
        try:
            return np.log2(args[0] / args[1])
        except:
            return np.nan
