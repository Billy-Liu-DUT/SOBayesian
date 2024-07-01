import numpy as np
import matplotlib.pyplot as plt



R = 8.314
Ea = 65 * 1e3
A = 5e8


time_range = np.linspace(10, 120, 100)
temp_range = np.linspace(25, 75, 100)
time_grid, temp_grid = np.meshgrid(time_range, temp_range)


yield_grid = np.zeros_like(time_grid)


for i in range(time_grid.shape[0]):
    for j in range(time_grid.shape[1]):
        T = temp_grid[i, j] + 273.15  # 转换为开尔文
        t = time_grid[i, j]


        k = A * np.exp(-Ea / (R * T))


        yield_grid[i, j] = (1 - np.exp(-k * t)) * 100


fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')

single_color = 'lightblue'


ax.plot_surface(time_grid, temp_grid, yield_grid, color=single_color)
plt.tight_layout(pad=3)


ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Yield (%)')

plt.savefig('reaction_yield_surface.png', format='png', bbox_inches='tight', dpi=500)
plt.show()