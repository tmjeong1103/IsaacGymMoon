import sys, os
sys.path.append('/home/yoonbyeong/Dev/AMP/IsaacGymMoon/IsaacGymJTM/isaacgym/python')
from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

motion_data_path = os.path.join(os.path.dirname(__file__),"data/walk_result_atlas.npy")
source_motion = SkeletonMotion.from_file(motion_data_path)
plot_skeleton_motion_interactive(source_motion)