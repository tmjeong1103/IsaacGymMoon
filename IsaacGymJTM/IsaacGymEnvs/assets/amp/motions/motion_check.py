import numpy as np

motion_path = 'cmu_walk_retarget_to_atlas.npy'
motion = np.load(motion_path,allow_pickle=True)

print(motion)