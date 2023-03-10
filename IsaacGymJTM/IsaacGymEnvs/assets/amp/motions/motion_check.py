import numpy as np
import os
motion_path = os.path.join(os.path.dirname(__file__), 'amp_humanoid_backflip.npy')
motion = np.load(motion_path,allow_pickle=True)

print(motion)