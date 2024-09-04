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
x_eq = np.linspace(-1, 1, n_points)
y_eq = np.linspace(-1, 1, n_points)
x1_eq, x2_eq = np.meshgrid(x_eq, y_eq)
z_eq = f2(x1_eq, x2_eq)

# Puntos de evaluación INTERMEDIOS (que no coincidan con los NODOS)
puntosIntermedios = np.linspace(-1, 1, 500)
x1_inter, x2_inter = np.meshgrid(puntosIntermedios, puntosIntermedios)
z_inter = f2(x1_inter, x2_inter)

# Interpolaciones
points = np.array([x1_eq.flatten(), x2_eq.flatten()]).T
values = z_eq.flatten()
grid_x, grid_y = np.mgrid[-1:1:500j, -1:1:500j]

# lineal
z_linear_interpol = griddata(points, values, (grid_x, grid_y), method='linear')

# cúbica
z_cubic_interpol = griddata(points, values, (grid_x, grid_y), method='cubic')

# Calcular el error absoluto
error_abs_linear = np.abs(z_inter - z_linear_interpol)
error_abs_cubic = np.abs(z_inter - z_cubic_interpol)

# Calcular el error relativo
error_rel_linear = error_abs_linear / np.abs(z_inter)
error_rel_cubic = error_abs_cubic / np.abs(z_inter)

""" GRÁFICOS 3D """
fig = plt.figure(figsize=(18, 6))

# f2
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x1_inter, x2_inter, z_inter, cmap='viridis')
ax1.set_title('Función Original')
pos1 = ax1.get_position()
ax1.set_position([pos1.x0 + 0.1, pos1.y0, pos1.width, pos1.height]) 

# lineal
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(grid_x, grid_y, z_linear_interpol, cmap='viridis')
ax2.set_title('Interpolación Lineal')
pos2 = ax2.get_position()
ax2.set_position([pos2.x0 + 0.05, pos2.y0, pos2.width, pos2.height])  # Mover 5 píxeles a la izquierda


# cúbica
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(grid_x, grid_y, z_cubic_interpol, cmap='viridis')
ax3.set_title('Interpolación Cúbica')
pos3 = ax3.get_position()
ax3.set_position([pos3.x0 + 0.05, pos3.y0, pos3.width, pos3.height])  # Mover 5 píxeles a la izquierda


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


""" VISUALIZACIÓN DEL ERROR EN 3D """

# Error absoluto
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(grid_x, grid_y, error_abs_linear, cmap='inferno')
ax1.set_title('Error Absoluto - Interpolación Lineal')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(grid_x, grid_y, error_abs_cubic, cmap='inferno')
ax2.set_title('Error Absoluto - Interpolación Cúbica')

plt.tight_layout()
plt.show()

# Error relativo
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(grid_x, grid_y, error_rel_linear, cmap='inferno')
ax1.set_title('Error Relativo - Interpolación Lineal')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(grid_x, grid_y, error_rel_cubic, cmap='inferno')
ax2.set_title('Error Relativo - Interpolación Cúbica')

plt.tight_layout()
plt.show()




# Números de nodos a evaluar
nodos = list(range(2, 51))

errores_rel_lineal = []
errores_rel_cubic = []

for n_points in nodos:
    x_eq = np.linspace(-1, 1, n_points)
    y_eq = np.linspace(-1, 1, n_points)
    x1_eq, x2_eq = np.meshgrid(x_eq, y_eq)
    z_eq = f2(x1_eq, x2_eq)
    
    # Interpolaciones
    points = np.array([x1_eq.flatten(), x2_eq.flatten()]).T
    values = z_eq.flatten()
    grid_x, grid_y = np.mgrid[-1:1:500j, -1:1:500j]
    
    # lineal
    z_linear_interpol = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # cúbica
    z_cubic_interpol = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    # Calcular el error relativo
    error_rel_linear = np.abs(z_inter - z_linear_interpol) / np.abs(z_inter)
    error_rel_cubic = np.abs(z_inter - z_cubic_interpol) / np.abs(z_inter)
    
    # Almacenar el error relativo mediano
    errores_rel_lineal.append(np.median(error_rel_linear))
    errores_rel_cubic.append(np.median(error_rel_cubic))

# Graficar los errores relativos
plt.figure(figsize=(10, 6))
plt.plot(nodos, errores_rel_lineal, label='Interpolación Lineal', marker='o', linestyle='-')
plt.plot(nodos, errores_rel_cubic, label='Interpolación Cúbica', marker='o', linestyle='--')
plt.xlabel('Número de Nodos')
plt.ylabel('Error Relativo Mediano')
plt.title('Error Relativo Mediano vs Número de Nodos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()