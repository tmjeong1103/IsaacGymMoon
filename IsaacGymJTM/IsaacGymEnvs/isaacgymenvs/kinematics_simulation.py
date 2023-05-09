"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Body physics properties example
-------------------------------
An example that demonstrates how to load rigid body, update its properties
and apply various actions. Specifically, there are three scenarios that
presents the following:
- Load rigid body asset with varying properties
- Modify body shape properties
- Modify body visual properties
- Apply body force
- Apply body linear velocity
"""
import time
from isaacgym import gymutil
from isaacgym import gymapi
import os
from tasks.amp.utils_amp.motion_lib import MotionLib
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch

# from tasks.common_rig_amp_base import KEY_BODY_NAMES
# initialize gym
gym = gymapi.acquire_gym()

KEY_BODY_NAMES = ["right_hand", "left_hand", "right_ankle", "left_ankle"]
# KEY_BODY_NAMES = ["r_hand", "l_hand", "r_foot", 'l_foot']

# parse arguments
args = gymutil.parse_arguments(description="Body Physics Properties Example")
device = 'cpu'
args.num_threads = 0
# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 3
spacing = 1.8
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create list to mantain environment and asset handles
envs = []
box_handles = []
actor_handles = []

# l5vd5
# create box assets w/ varying densities (measured in kg/m^3)
box_size = 0.2
box_densities = [8., 32., 1024.]

# create env
env = gym.create_env(sim, env_lower, env_upper, 1)
envs.append(env)

name = 'common rig'
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
# asset_file = "mjcf/atlas_v5_m/atlas_v5_strong.xml"
asset_file = "mjcf/common_rig_strong.xml"
asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
asset_options.angular_damping = 0.05
asset_options.max_angular_velocity = 100.0
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
actor_handles.append(gym.create_actor(env, humanoid_asset, gymapi.Transform(p=gymapi.Vec3(0., 0.0, 0.)), name, 0, 0))
# look at the first env
cam_pos = gymapi.Vec3(8, -0, 3)
cam_target = gymapi.Vec3(-0.8, 2, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
num_bodies = gym.get_asset_rigid_body_count(humanoid_asset)# + self.gym.get_asset_rigid_body_count(ball_asset) + self.gym.get_asset_rigid_body_count(box_asset)
num_dof = gym.get_asset_dof_count(humanoid_asset)
num_joints = gym.get_asset_joint_count(humanoid_asset)

motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/amp/motions/333.npy")
# motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/amp/motions/12_01_walk.npy")

body_ids = []
for body_name in KEY_BODY_NAMES:
    body_id = gym.find_actor_rigid_body_handle(env, actor_handles[0], body_name)
    assert(body_id != -1)
    body_ids.append(body_id)

body_ids = to_torch(body_ids, device=device, dtype=torch.long)

motion_lib = MotionLib(motion_file=motion_file_path, 
                                num_dofs=num_dof,
                                key_body_ids=body_ids.cpu().numpy(),
                                device=device)
motion_ids = motion_lib.sample_motions(1)
motion_times = 0
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
_dof_state = gymtorch.wrap_tensor(dof_state_tensor)
_dof_pos = _dof_state.view(1, num_dof, 2)[..., 0]
_dof_vel = _dof_state.view(1, num_dof, 2)[..., 1]
actor_indices = torch.arange(1, dtype=torch.long, device=device).view(1, 1)
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
_root_states = gymtorch.wrap_tensor(actor_root_state)#[::2].contiguous()
while not gym.query_viewer_has_closed(viewer):
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = motion_lib.get_motion_state(motion_ids, motion_times)
    # root_pos[:,2] += 1.2

    _root_states[0, 0:3] = root_pos
    _root_states[0, 3:7] = root_rot
    _root_states[0, 7:10] = root_vel
    _root_states[0, 10:13] = root_ang_vel
    # dof_pos[0, 4] = 1.5629
    # dof_pos[0, 5] = -0.2429
    # dof_pos[0, 6] = 0.0451
    # dof_pos[0, 7] = 0.5138
    # dof_pos[0, 8] = 0.3380
    # dof_pos[0, 9] = -0.6591
    # dof_pos[0, 10] = -0.3536
    # dof_pos = torch.zeros_like(dof_pos)
    # dof_pos[0, -1] = 1.5
    _dof_pos[0] = dof_pos
    _dof_vel[0] = dof_vel

    # Added from JTM
    env_ids_int32 = actor_indices[0].to(dtype=torch.int32)

    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(_root_states), 
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(_dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    # step the physics
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    # gym.simulate(sim)
    time.sleep(0.05)
    motion_times += 1/20


print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
