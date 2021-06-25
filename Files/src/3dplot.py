import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x   = np.linspace(1,5,100)
y1  = np.ones(x.size)
y2  = np.ones(x.size)*2
y3  = np.ones(x.size)*3
z   = np.sin(x/2)

pl.figure()
ax = pl.subplot(projection='3d')
ax.plot(x, y1, z, color='r')
ax.plot(x, y2, z, color='g')
ax.plot(x, y3, z, color='b')

ax.add_collection3d(pl.fill_between(x, 0.95*z, 1.05*z, color='r', alpha=0.3), zs=1, zdir='y')
ax.add_collection3d(pl.fill_between(x, 0.90*z, 1.10*z, color='g', alpha=0.3), zs=2, zdir='y')
ax.add_collection3d(pl.fill_between(x, 0.85*z, 1.15*z, color='b', alpha=0.3), zs=3, zdir='y')

ax.set_xlabel('Day')
ax.set_zlabel('Resistance (%)')

pl.show()