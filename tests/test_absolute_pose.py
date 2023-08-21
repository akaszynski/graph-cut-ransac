import os
import numpy as np
import pytest
import pygcransac
import random
from test_utils import calculate_error

# Define paths to data files (you can adjust these paths as needed)
THIS_PATH = os.path.dirname(os.path.abspath(__file__))

CORRESPONDENCES_PATH = os.path.join(THIS_PATH, '..', 'examples/img/pose6dscene_points.txt')
GT_POSE_PATH = os.path.join(THIS_PATH, '..', 'examples/img/pose6dscene_gt.txt')
INTRINSIC_PARAMS_PATH = os.path.join(THIS_PATH, '..', 'examples/img/pose6dscene.K')

# Load data from files
GT_POSE = np.loadtxt(GT_POSE_PATH)
INTRINSIC_PARAMS = np.loadtxt(INTRINSIC_PARAMS_PATH)


@pytest.fixture
def normalized_correspondences():

    def normalize_image_points(corrs, K): 
        n = len(corrs)
        normalized_correspondences = np.zeros((corrs.shape[0], 5))
        inv_K = np.linalg.inv(K)

        for i in range(n):
            p1 = np.append(correspondences[i][0:2], 1)
            p2 = inv_K.dot(p1)
            normalized_correspondences[i][0:2] = p2[0:2]
            normalized_correspondences[i][2:] = correspondences[i][2:]
        return normalized_correspondences


    correspondences = np.loadtxt(CORRESPONDENCES_PATH)
    outlier_number = round(3.0 * correspondences.shape[0])
    mins = np.min(correspondences, axis=0)
    maxs = np.max(correspondences, axis=0)

    mins[0] = 0
    mins[1] = 0

    outliers = []
    for i in range(outlier_number):
        for dim in range(5):
            outliers.append(random.uniform(mins[dim], maxs[dim]))
    outliers = np.array(outliers).reshape(-1, 5)
    correspondences = np.concatenate((correspondences, outliers))
    return normalize_image_points(correspondences, INTRINSIC_PARAMS)


@pytest.mark.flaky(reruns=3)  # sometimes fails to find pose
def test_pygcransac(normalized_correspondences):
    threshold = 2.0
    normalized_threshold = threshold / (INTRINSIC_PARAMS[0, 0] + INTRINSIC_PARAMS[1, 1]) / 2.0
    pose, mask = pygcransac.find6DPose(
        np.ascontiguousarray(normalized_correspondences),
        min_iters = 50,
        max_iters = 1000,
        probabilities = [],
        sampler = 0,
        threshold = normalized_threshold,
        conf = 0.99)
    if pose is None:
        raise RuntimeError('Failed to compute pose')

    err_R, err_t = calculate_error(GT_POSE, pose)
    assert err_R < 0.01  # rotation error less than 0.01 degrees
    assert err_t < 0.001  # translation error less than 0.001 mm
