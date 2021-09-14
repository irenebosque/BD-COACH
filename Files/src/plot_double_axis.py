import matplotlib.pyplot as plt

fig, ax  = plt.subplots(1)

main_x_axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
addi_x_axis = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

y1_values = [0, 0.1, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1, 1, 1]
y2_values = [0, 0.2, 0.3, 0.4, 0.75, 0.9, 0.95, 0.9, 1, 1, 1]
y3_values = [0, 0.15, 0.2, 0.3, 0.8, 0.75, 0.7, 1, 0.8, 0.9, 0.95]

tests = [y1_values, y2_values, y3_values]


ax2 = ax.twiny()

for test in tests:
    ax.plot(main_x_axis, test)






    ax2.spines["bottom"].set_position(("axes", -.1)) # move it down
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')


    ax2.plot(addi_x_axis,test,  alpha=0) #redraw to put twiny on same scale



plt.show()