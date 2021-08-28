# Copyright (c) Gorilla-Lab. All rights reserved.
import math

import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import transforms3d.euler as euler

def elastic(xyz, gran, mag):
    """Elastic distortion (from point group)

    Args:
        xyz (np.ndarray): input point cloud
        gran (float): distortion param
        mag (float): distortion scalar

    Returns:
        xyz: point cloud with elastic distortion
    """
    blur0 = np.ones((3, 1, 1)).astype("float32") / 3
    blur1 = np.ones((1, 3, 1)).astype("float32") / 3
    blur2 = np.ones((1, 1, 3)).astype("float32") / 3

    bb = np.abs(xyz).max(0).astype(np.int32)//gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
    noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(xyz_):
        return np.hstack([i(xyz_)[:,None] for i in interp])
    return xyz + g(xyz) * mag


# modify from PointGroup
def pc_aug(xyz, scale=False, flip=False, rot=False):
    if scale:
        scale = np.random.uniform(0.8, 1.2)
        xyz = xyz * scale
    if flip:
        # m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        flag = np.random.randint(0, 2)
        if flag:
            xyz[:, 0] = -xyz[:, 0]
    if rot:
        theta = np.random.uniform() * np.pi
        # theta = np.random.randn() * np.pi
        rot_mat = np.eye(3)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat[0, 0] = c
        rot_mat[0, 1] = -s
        rot_mat[1, 1] = c
        rot_mat[1, 0] = s
        xyz = xyz @ rot_mat.T

    return xyz



