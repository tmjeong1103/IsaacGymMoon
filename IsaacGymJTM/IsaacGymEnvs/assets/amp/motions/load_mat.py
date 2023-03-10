import numpy as np 
import matplotlib.pylab as plt 
import scipy.io
import os
import sys
import torch

sys.path.append('/home/yoonbyeong/Dev/AMP/IsaacGymMoon/IsaacGymJTM/IsaacGymEnvs/isaacgymenvs/tasks/amp/poselib')
from collections import OrderedDict
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

mat_file_name =  "07_01_walk_result.mat"
mat_file_path = os.path.join(os.path.dirname(__file__), mat_file_name)
mat_file = scipy.io.loadmat(mat_file_path)

npy_file_name = 'walk_16hz_trunc.npy'
npy_file = np.load(os.path.join(os.path.dirname(__file__), npy_file_name), allow_pickle=True)
skel_file = np.load(os.path.join(os.path.dirname(__file__), 'atlas_zero_pose.npy'), allow_pickle=True)
# npy_file_name2 = 'post_process2.npy'
# npy_file2 = np.load(os.path.join(os.path.dirname(__file__), npy_file_name2), allow_pickle=True)

npy_file.item()['rotation']['arr'] = np.array(mat_file['rotation'])
npy_file.item()['root_translation']['arr'] = np.array(mat_file['root_translation'])
npy_file.item()['skeleton_tree'] = skel_file.item()['skeleton_tree']#np.array(mat_file['skel_local_translation'])
npy_file.item()['fps'] = 60

global_velocity = SkeletonMotion._compute_velocity(p=torch.tensor(mat_file['global_translation']), time_delta=1/60)
global_angular_velocity = SkeletonMotion._compute_angular_velocity(r=torch.tensor(mat_file['global_rotation']), time_delta=1/60)
npy_file.item()['global_velocity']['arr'] = np.array(global_velocity)
npy_file.item()['global_angular_velocity']['arr'] = np.array(global_angular_velocity)

npy_file
mat_file

# npy_file.item()['rotation']['arr'] = npy_file.item()['rotation']['arr'][30:]
# npy_file.item()['root_translation']['arr'] = npy_file.item()['root_translation']['arr'][30:]
# npy_file.item()['global_velocity']['arr'] = npy_file.item()['global_velocity']['arr'][30:]
# npy_file.item()['global_angular_velocity']['arr'] = npy_file.item()['global_angular_velocity']['arr'][30:]
# np.array(OrderedDict([('rotation', {'arr'})]))
# npy_file.item()[''] = mat_file['global_rotation']