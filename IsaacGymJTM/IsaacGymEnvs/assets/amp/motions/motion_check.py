import numpy as np

motion_path = 'cmu_run_motion_atlas.npy'
motion = np.load(motion_path,allow_pickle=True)

print(motion)