import numpy as np 
import matplotlib.pylab as plt 
import scipy.io
import os
import sys
import torch
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

sys.path.append('/home/yoonbyeong/Dev/AMP/IsaacGymMoon/IsaacGymJTM/IsaacGymEnvs/isaacgymenvs/tasks/amp/poselib')
from collections import OrderedDict
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

mat_file_name =  "02_05_punch_strike_result.mat"
mat_file_path = os.path.join(os.path.dirname(__file__), mat_file_name)
mat_file = scipy.io.loadmat(mat_file_path)

data = mat_file['global_translation']

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Init scat
scats = [ax.scatter(dat[:,0], dat[:,1], dat[:,2]) for dat in data[0]]
ax.xlim
print(data.shape)
plt.title('Motion Joint')

def update_scats(num, data, scats):
    for scat, dat in zip(scats, data):
        # NOTE: there is no .set_data() for 3 dim data...
        scat._offsets3d = (data[num,:,0], data[num,:,1], data[num,:,2])

        # scat.set_data(dat[num, 0:2])
        # scat.set_3d_properties(dat[num, 2])
    return scat


scat_ani = animation.FuncAnimation(fig, update_scats, len(data), fargs=(data, scats),
                                   interval=1, blit=False)

plt.show()