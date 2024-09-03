import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def f2(x1, x2):
    term1 = 0.75 * np.exp(-((9 * x1 - 2)**2 / 4) - ((9 * x2 - 2)**2 / 4))
    term2 = 0.75 * np.exp(-((9 * x1 + 1)**2 / 49) - ((9 * x2 + 1)**2 / 10))
    term3 = 0.5 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    term4 = -0.2 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    
    return term1 + term2 + term3 + term4

# Puntos equiespaciados
n_points = 20
i = np.arange(n_points)
x_chebyshev = np.cos((2 * i + 1) / (2 * n_points) * np.pi)
y_chebyshev = np.cos((2 * i + 1) / (2 * n_points) * np.pi)

# Crear malla de puntos de Chebyshev
x_chebyshev, y_chebyshev = np.meshgrid(x_chebyshev, y_chebyshev)
z_eq = f2(x_chebyshev, y_chebyshev)


# Puntos de evaluación INTERMEDIOS (que no coincidan con los NODOS)
puntosIntermedios = np.linspace(-1, 1, 500)
x1_inter, x2_inter = np.meshgrid(puntosIntermedios, puntosIntermedios)
z_inter = f2(x1_inter, x2_inter)

# Interpolaciones
points = np.array([x_chebyshev.flatten(), y_chebyshev.flatten()]).T
values = z_eq.flatten()
grid_x, grid_y = np.mgrid[-1:1:500j, -1:1:500j]

# lineal
z_linear_interpol = griddata(points, values, (grid_x, grid_y), method='linear')

# cúbica
z_cubic_interpol = griddata(points, values, (grid_x, grid_y), method='cubic')

""" GRÁFICOS 3D """
fig = plt.figure(figsize=(18, 6))

# f2
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x1_inter, x2_inter, z_inter, cmap='viridis')
ax1.set_title('Función Original')

# lineal
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(grid_x, grid_y, z_linear_interpol, cmap='viridis')
ax2.set_title('Interpolación Lineal')

# cúbica
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(grid_x, grid_y, z_cubic_interpol, cmap='viridis')
ax3.set_title('Interpolación Cúbica')

plt.tight_layout()
plt.show()

""" GRÁFICOS """
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1) # Función original
plt.contourf(x1_inter, x2_inter, z_inter, cmap='viridis')
plt.title('Función Original')
plt.colorbar()

plt.subplot(1, 3, 2) # Interpolación lineal
plt.contourf(grid_x, grid_y, z_linear_interpol, cmap='viridis')
plt.title('Interpolación Lineal')
plt.colorbar()

plt.subplot(1, 3, 3) # Interpolación cúbica
plt.contourf(grid_x, grid_y, z_cubic_interpol, cmap='viridis')
plt.title('Interpolación Cúbica')
plt.colorbar()

plt.tight_layout()
plt.show()


"""
Observaciones: funciona peor con puntos no equiespaciados

HACER: ver como llevar cuenta del error más allá de verlo graficamente
"""


"""
meshgrid 
griddata #usar

PUNTO 2
trabajar con curva parametrica en q f va de r a r3 (de tiempo a pos)
entonces f(T) = x(t) y(t) z(t) pero cada una ve de r en r
"""


