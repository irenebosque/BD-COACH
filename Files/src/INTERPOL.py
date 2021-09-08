import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import pandas as pd


def postProcess(test_name):
    path = './results/'

    timesteps_list = []
    return_list = []

    d_frame = pd.read_csv(path + test_name)

    frame_to_list = d_frame.values.tolist()

    del frame_to_list[0][0]
    del frame_to_list[14][0]

    timesteps_list.append(frame_to_list[0])
    return_list.append(frame_to_list[14])

    return timesteps_list, return_list

x, y = postProcess('DCOACH_HM-True_e-0.1_B-10006_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-12.csv')
print("x:", len(x[0]))
print("y:", len(y[0]))

f = interpolate.interp1d(x[0], y[0], fill_value="extrapolate")

xnew = np.arange(0, 40000, 10)
print('xnew: ', xnew)

ynew = f(xnew)   # use interpolation function returned by `interp1d`
print('ynew: ', ynew)

import pandas as pd

data = {'xnew': xnew,
        'ynew': ynew
        }

df = pd.DataFrame(data, columns= ['xnew', 'ynew'])

print (df)
df.to_csv('./results/test12.csv')



# plt.plot(x, y, xnew, ynew)
# plt.ylim(0, 1.2)
# plt.xlim(0, 40000)
#
# plt.show()