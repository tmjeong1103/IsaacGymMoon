import numpy as np
import os

motion_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'atlas_walk_post_process.npy')
motion = np.load(motion_path,allow_pickle=True)

motion2_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kickoff_walk.npy')
motion2 = np.load(motion2_path,allow_pickle=True)

print(motion)