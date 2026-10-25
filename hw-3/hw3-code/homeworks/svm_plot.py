import numpy as np
import matplotlib.pyplot as plt

red_points = np.array([[1, 2], [1, 3], [2, 3], [3, 4]])
blue_points = np.array([[0, 0.5], [1, 0], [2, 1], [3, 0]])

red_x, red_y = red_points[:, 0], red_points[:, 1]
blue_x, blue_y = blue_points[:, 0], blue_points[:, 1]

plt.scatter(red_x, red_y, color='red', label='$y=-1$')
plt.scatter(blue_x, blue_y, color='blue', label='$y=1$')

x_range = np.linspace(0, 3, 4)
line_y1 = 0.25 * x_range + 1.75
line_y2 = 0.25 * x_range + 0.5
line_y3 = 0.25 * x_range + 1.125

plt.plot(x_range, line_y1, color='red', label='Hyperplane')
plt.plot(x_range, line_y2, color='blue', label='Hyperplane')
plt.plot(x_range, line_y3, color='black', label='Maximum-margin hyperplane')

plt.scatter(red_x[0], red_y[0], color='white', marker='o', s=200, edgecolors='purple', linewidths=2, label='Support vectors')
plt.scatter(blue_x[0], blue_y[0], color='white', marker='o', s=200, edgecolors='purple', linewidths=2)
plt.scatter(blue_x[2], blue_y[2], color='white', marker='o', s=200, edgecolors='purple', linewidths=2)
plt.scatter(red_x, red_y, color='red')
plt.scatter(blue_x, blue_y, color='blue')

plt.arrow(1.5, 1.5, 0.05, -0.4, head_width=0.05, head_length=0.1, linewidth=4, color='green', label='$w$')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

plt.show()