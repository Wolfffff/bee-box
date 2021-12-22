import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tqdm
from scipy.signal import savgol_filter


def fill_gaps(x, max_len):
    """Dilate -> erode"""
    return scipy.ndimage.binary_closing(x, structure=np.ones((max_len,)))


def connected_components1d(x, return_limits=False):
    """Return the indices of the connected components in a 1D logical array.
    
    Args:
        x: 1d logical (boolean) array.
        return_limits: If True, return indices of the limits of each component rather
            than every index. Defaults to False.
            
    Returns:
        If return_limits is False, a list of (variable size) arrays are returned, where
        each array contains the indices of each connected component.
        
        If return_limits is True, a single array of size (n, 2) is returned where the
        columns contain the indices of the starts and ends of each component.
    """
    L, n = scipy.ndimage.label(x.squeeze())
    ccs = scipy.ndimage.find_objects(L)
    starts = [cc[0].start for cc in ccs]
    ends = [cc[0].stop for cc in ccs]
    if return_limits:
        return np.stack([starts, ends], axis=1)
    else:
        return [np.arange(i0, i1, dtype=int) for i0, i1 in zip(starts, ends)]


def flatten_features(x, axis=0):
    
    if axis != 0:
        # Move time axis to the first dim
        x = np.moveaxis(x, axis, 0)
    
    # Flatten to 2D.
    initial_shape = x.shape
    x = x.reshape(len(x), -1)
    
    return x, initial_shape


def unflatten_features(x, initial_shape, axis=0):
    # Reshape.
    x = x.reshape(initial_shape)
    
    if axis != 0:
        # Move time axis back
        x = np.moveaxis(x, 0, axis)
        
    return x


def fill_missing(x, kind="nearest", axis=0, **kwargs):
    """Fill missing values in a timeseries.
    
    Args:
        x: Timeseries of shape (time, ...) or with time axis specified by axis.
        kind: Type of interpolation to use. Defaults to "nearest".
        axis: Time axis (default: 0).
    
    Returns:
        Timeseries of the same shape as the input with NaNs filled in.
    
    Notes:
        This uses pandas.DataFrame.interpolate and accepts the same kwargs.
    """
    if x.ndim > 2:
        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)
        
        # Interpolate.
        x = fill_missing(x, kind=kind, axis=0, **kwargs)
        
        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        
        return x
        
    return pd.DataFrame(x).interpolate(method=kind, axis=axis, **kwargs).to_numpy()


def smooth_ewma(x, alpha, axis=0, inplace=False):        
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()
            
        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)
        
        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_ewma(x[:, i], alpha, axis=0)
        
        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x
    initial_shape = x.shape
    x = pd.DataFrame(x).ewm(alpha=alpha, axis=axis).mean().to_numpy()
    return x.reshape(initial_shape) 


def smooth_gaussian(x, std=1, window=5, axis=0, inplace=False):        
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()
            
        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)
        
        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_gaussian(x[:, i], std, window, axis=0)
        
        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x
    
    y = pd.DataFrame(x.copy()).rolling(window, win_type="gaussian", center=True).mean(std=std).to_numpy()
    y = y.reshape(x.shape)
    mask = np.isnan(y) & (~np.isnan(x))
    y[mask] = x[mask]
    return y


def smooth_median(x, window=5, axis=0, inplace=False):        
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()
            
        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)
        
        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_median(x[:, i], window, axis=0)
        
        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x
    
    y = scipy.signal.medfilt(x.copy(), window)
    y = y.reshape(x.shape)
    mask = np.isnan(y) & (~np.isnan(x))
    y[mask] = x[mask]
    return y


def update_labels(labels, trx, copy=False):
    """Update a sleap.Labels from a tracks array.
    
    Args:
        labels: sleap.Labels
        trx: numpy array of shape (frames, tracks, nodes, 2)
        copy: If True, make a copy of the labels rather than updating in place.
    
    Return:
        The labels with updated points.
    """
    import sleap
    if copy:
        labels = labels.copy()
    
    for i in range(len(trx)):
        lf = labels[i]
        for j in range(trx.shape[1]):
            track_j = labels.tracks[j]
            updated_j = False
            for inst in lf:
                if inst.track != track_j:
                    continue
                for node, pt in zip(labels.skeleton.nodes, trx[i, j]):
                    if np.isnan(pt).all():
                        continue
                    inst[node].x, inst[node].y = pt
                updated_j = True
            if not updated_j and not np.isnan(trx[i, j]).all():
                inst = sleap.Instance.from_numpy(trx[i, j], labels.skeleton, track=track_j)
                inst.frame = lf
                lf.instances.append(inst)
    
    return labels


def normalize_to_egocentric(x, rel_to=None, scale_factor=1, ctr_ind=1, fwd_ind=0, fill=True, return_angles=False):
    """Normalize pose estimates to egocentric coordinates.
    
    Args:
        x: Pose of shape (joints, 2) or (time, joints, 2)
        rel_to: Pose to align x with of shape (joints, 2) or (time, joints, 2). Defaults
            to x if not specified.
        scale_factor: Spatial scaling to apply to coordinates after centering.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.
        fill: If True, interpolate missing ctr and fwd coordinates. If False, timesteps
            with missing coordinates will be all NaN. Defaults to True.
        return_angles: If True, return angles with the aligned coordinates.
    
    Returns:
        Egocentrically aligned poses of the same shape as the input.
        
        If return_angles is True, also returns a vector of angles.
    """
        
    if rel_to is None:
        rel_to = x
    
    is_singleton = (x.ndim == 2) and (rel_to.ndim == 2)
    
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    if rel_to.ndim == 2:
        rel_to = np.expand_dims(rel_to, axis=0)
    
    # Find egocentric forward coordinates.
    ctr = rel_to[..., ctr_ind, :]  # (t, 2)
    fwd = rel_to[..., fwd_ind, :]  # (t, 2)
    if fill:
        ctr = fill_missing(ctr, kind="nearest")
        fwd = fill_missing(fwd, kind="nearest")
    ego_fwd = fwd - ctr
    
    # Compute angle.
    ang = np.arctan2(ego_fwd[..., 1], ego_fwd[..., 0])  # arctan2(y, x) -> radians in [-pi, pi]
    ca = np.cos(ang)  # (t,)
    sa = np.sin(ang)  # (t,)
    
    # Build rotation matrix.
    rot = np.zeros([len(ca), 3, 3], dtype=ca.dtype)
    rot[..., 0, 0] = ca
    rot[..., 0, 1] = -sa
    rot[..., 1, 0] = sa
    rot[..., 1, 1] = ca
    rot[..., 2, 2] = 1
    
    # Center and scale.
    x = x - np.expand_dims(ctr, axis=1)
    x /= scale_factor
    
    # Pad, rotate and crop.
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1) @ rot
    x = x[..., :2]
    
    if is_singleton:
        x = x[0]
    
    if return_angles:
        return x, ang
    else:
        return x

def instance_node_velocities(fly_node_locations,start_frame,end_frame):
    frame_count = len(range(start_frame, end_frame))
    if len(fly_node_locations.shape) == 4:
        for fly_idx in range(fly_node_locations.shape[3]):
            fly_node_velocities = np.zeros((frame_count,fly_node_locations.shape[1],fly_node_locations.shape[3] ))
            for n in tqdm(range(0, fly_node_locations.shape[1])):
                fly_node_velocities[:, n,fly_idx] = smooth_diff(fly_node_locations[start_frame:end_frame, n, :,fly_idx])
    else:
        fly_node_velocities = np.zeros((frame_count,fly_node_locations.shape[1]))
        for n in tqdm(range(0, fly_node_locations.shape[1] - 1)):
            fly_node_velocities[:, n] = smooth_diff(fly_node_locations[start_frame:end_frame, n, :])

    return fly_node_velocities


def instance_node_velocities_bees(fly_node_locations,start_frame,end_frame):
    frame_count = len(range(start_frame, end_frame))
    fly_node_velocities = np.zeros((frame_count,fly_node_locations.shape[2]))
    for bee in tqdm(range(0, fly_node_locations.shape[2])):
        fly_node_velocities[:, bee] = smooth_diff(fly_node_locations[start_frame:end_frame, :, bee])
    return fly_node_velocities

def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] arrayF
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    node_loc_vel = np.zeros_like(node_loc)
    
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel

from scipy.interpolate import interp1d
from tqdm import tqdm


def fill_missing_np(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # Interpolate along each slice. 
    for i in tqdm(range(Y.shape[-1])):
        try:
            y = Y[:, i]

            # Build interpolant.
            x = np.flatnonzero(~np.isnan(y))
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)
            
            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

            # Save slice
            Y[:, i] = y
        except:
            Y[:, i] = 0
    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def hist_sort(locations,ctr_idx, xbins=2, ybins=2, xmin=0, xmax=1536, ymin=0, ymax=1536):
    assignments = []
    freq = []
    for track_num in range(locations.shape[3]):
        h, xedges, yedges = np.histogram2d(
            locations[:, ctr_idx, 0, track_num],
            locations[:,ctr_idx, 1, track_num],
            range=[[xmin, xmax], [ymin, ymax]],
            bins=[xbins, ybins],
        )
        # Probably not the correct way to flip this around to get rows and columns correct but it'll do!
        assignment = h.argmax()
        freq.append(h)
        assignments.append(assignment)
    assignment_indices = np.argsort(assignments)
    locations = locations[:, :, :, assignment_indices]
    return assignment_indices, locations, freq

import scipy.stats
def hist_sort_rowwise(locations,ctr_idx, xbins=2, ybins=2, xmin=0, xmax=1536, ymin=0, ymax=1536):
    fourbyfour_interior_gridmap = {5:0,6:1,9:2,10:3}
    output = np.zeros_like(locations)
    for track_num in range(locations.shape[3]):
        h, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(
            locations[:, ctr_idx, 0, track_num],
            locations[:,ctr_idx, 1, track_num],
            None,
            'count',
            range=[[xmin, xmax], [ymin, ymax]],
            bins=[xbins, ybins],
        )
        assignments = np.vectorize(fourbyfour_interior_gridmap.get)(binnumber)
        frames,nodes,coords,assignments = np.arange(locations.shape[0]), slice(None),slice(None),assignments
        output[frames,nodes,coords,assignments] = locations[frames,nodes,coords,assignments]
    return output

def acorr(x, normed=True, maxlags=10):
    return xcorr(x,x, normed, maxlags)

def xcorr(x, y, normed=True, maxlags=10):

        Nx = len(x)
        if Nx != len(y):
            raise ValueError('x and y must be equal length')

        correls = np.correlate(x, y, mode="full")

        if normed:
            correls /= np.sqrt(np.dot(x, x) * np.dot(y, y))

        if maxlags is None:
            maxlags = Nx - 1

        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maxlags must be None or strictly '
                             'positive < %d' % Nx)
        lags = np.arange(-maxlags, maxlags + 1)
        correls = correls[Nx - 1 - maxlags:Nx + maxlags]
        return lags, correls

def isintimeperiod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        return nowTime >= startTime or nowTime <= endTime