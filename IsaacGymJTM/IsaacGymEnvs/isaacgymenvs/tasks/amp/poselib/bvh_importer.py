import os
import json
from bvh import Bvh
import numpy as np
import math
import torch
from poselib.core.rotation3d import quat_mul, quat_from_angle_axis

from poselib.skeleton.backend.bvh.bvh_backend import bvh_to_array
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

deg2rad = np.pi / 180.0

# Subject and mocap number
subject = str(10)
clip = str(4)
subject_str = (3 - len(subject)) * "0" + subject
subject_short = (2 - len(subject)) * "0" + subject
clip_str = (2 - len(clip)) * "0" + clip

# workspace base path identification
base = "{}/data/bvh/".format(os.getcwd())
bvh_file = "{}/data/bvh/02_03.bvh".format(os.getcwd())
#bvh_file = os.path.abspath("{}/{}/{}_{}.bvh".format(base,subject_str,subject_short,clip_str))

root_joint = "Hips"

source_tpose = SkeletonState.from_file("data/cmu_tpose.npy")
target_tpose = SkeletonState.from_file("data/atlas_tpose.npy")

#a,b,c,d,e = bvh_to_array(bvh_file,root_joint="Hips",debug=True)

motion = SkeletonMotion.from_bvh(
    bvh_file_path=bvh_file,
    root_joint="Hips",
    channels = ["Xrotation", "Yrotation","Zrotation"],
    root_xyz_channels = ["Xposition","Yposition","Zposition"],
    offset_order=[0,2,1],
#    skeleton_tree=source_tpose.skeleton_tree
)

print(motion.skeleton_tree)
print(source_tpose.skeleton_tree)

#print(motion.skeleton_state)

zero_pose = SkeletonState.zero_pose(motion.skeleton_tree)

ref = zero_pose.rotation[0,:]
n = np.array([1/math.sqrt(2),0.0,0.0,1/math.sqrt(2)])

rotation_quats = torch.from_numpy(n)

zero_pose.rotation[0,:]= quat_mul(rotation_quats, ref)

zero_pose.root_translation[:] = zero_pose.root_translation + torch.from_numpy(np.array([0.0,0.0,17.5]))
lrot = quat_from_angle_axis(torch.Tensor([-20.0]), torch.Tensor([0.0,0.0,1.0]),True)
# 1 = Lefthipjoint, 7 = Righthipjoint
rrot = quat_from_angle_axis(torch.Tensor([20.0]), torch.Tensor([0.0,0.0,1.0]),True)

#zero_pose.rotation[2,:] = quat_mul(torch.from_numpy(np.array([0.0,1/math.sqrt(2),0.0,1/math.sqrt(2)])),zero_pose.rotation[2,:])
zero_pose.rotation[2,:] = quat_mul(lrot, zero_pose.rotation[2,:])
zero_pose.rotation[7,:] = quat_mul(rrot, zero_pose.rotation[7,:])

print(zero_pose.root_translation)
print(zero_pose.rotation)
print(len(zero_pose.local_transformation))
print(zero_pose.num_joints)

plot_skeleton_state(zero_pose)
plot_skeleton_state(source_tpose)
plot_skeleton_state(target_tpose)

zero_pose.to_file("data/cmu_zero_pose.npy")
motion.to_file("data/cmu_run_motion.npy")

plot_skeleton_motion_interactive(motion)