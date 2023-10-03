#!/usr/bin/python
#encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation 
import MG

xdata, ydata = [], []
cont = 0
frames = np.linspace(0, 8*np.pi, 1024)

fig = plt.figure(figsize=(3,3))
ax =p3.Axes3D(fig)

N=1000
mg = MG.MackeyGlass(10, init=[0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759])
itMG = iter(mg)

data=np.zeros((3, N))
for i in range(N):
   data[0, i] = next(itMG)
for i in range(500):
   next(itMG)
for i in range(N):
   data[1, i] = next(itMG)
for i in range(500):
   next(itMG)
for i in range(N):
   data[2, i] = next(itMG)
[minX, maxX] = [np.min(data[0,:]),np.max(data[0,:])]
[minY, maxY] = [np.min(data[1,:]),np.max(data[1,:])]
[minZ, maxZ] = [np.min(data[2,:]),np.max(data[2,:])]
lines = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])[0]
ax.set_xlim3d([1.1*minX, 1.1*maxX])
ax.set_ylim3d([1.1*minY, 1.1*maxY])
ax.set_zlim3d([1.1*minZ, 1.1*maxZ])


def update(num, data, line):
   line.set_data(data[0:2, (num-20):num])
   line.set_3d_properties(data[2,(num-20):num])
   return line

ani = animation.FuncAnimation(fig, update, len(frames), fargs=(data, lines), blit=False, interval=10)
plt.show()
