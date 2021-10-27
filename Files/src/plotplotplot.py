#Use the extended slicing syntax list[start:stop:step] to get a new list containing every nth element. Leave start and stop empty, and set step to the desired n.

a_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

first_part = a_list[0:5:1]
second_part = a_list[5::2]

complete = first_part + second_part
print(complete)

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np

x = np.linspace(0, 10, 1000)
y = 0.000001 * np.sin(10 * x)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y)


def y_fmt(x, y):
    return '{:2.2e}'.format(x).replace('e', 'x10^')


ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
