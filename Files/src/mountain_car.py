from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

timesteps = [[0, 501, 1001, 1501, 2001, 2501, 3001, 3052, 3552, 4052, 4552, 5052, 5552, 5650, 6150, 6650, 7150 ,7650, 8150, 8249]]
success = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]
timesteps_list = []
success_list = []


for t, s in zip(timesteps, success):
    print('t: ', t)
    print('s: ', s)

    fun_r = interp1d(t, s)

    timesteps_last_episode = t[-1]
    timesteps_last_episode = np.int64(timesteps_last_episode)

    xnew = np.linspace(0, timesteps_last_episode, num=timesteps_last_episode, endpoint=False)
    xnew = xnew.astype(int)


    timesteps_list.append(xnew)
    success_list.append(fun_r(xnew))



timesteps_list = timesteps_list[0]
success_list = success_list[0]


timesteps_list = timesteps_list.tolist()
success_list = success_list.tolist()




fig, ax1 = plt.subplots(1)
ax1.plot(timesteps_list, success_list)
ax1.grid(linestyle='--')
plt.show()


