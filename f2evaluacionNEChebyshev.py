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
ax2.set_position([pos2.x0 + 0.05, pos2.y0, pos2.width, pos2.height])

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




""" HACER ERRORES RELATIVOS COMPARATIVOS"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# Definición de la función continua f2
def f2(x):
    term1 = 0.75 * np.exp(-((9 * x - 2)**2 / 4))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2 / 49))
    term3 = 0.5 * np.exp(-((9 * x - 7)**2 / 4))
    term4 = -0.2 * np.exp(-((9 * x - 7)**2 / 4))
    return term1 + term2 + term3 + term4

# Números de nodos a evaluar
nodos = list(range(2, 51))  # números de nodos a evaluar
errores_lineal = []
errores_cubica = []

# Puntos de evaluación intermedios (dentro del rango de los nodos de Chebyshev)
puntos_intermedios = np.linspace(-1, 1, 500)
z_inter = f2(puntos_intermedios)

for n_points in nodos:
    # Calcular nodos de Chebyshev
    x_chebyshev = np.cos((2 * np.arange(n_points) + 1) / (2 * n_points) * np.pi)
    y_chebyshev = f2(x_chebyshev)
    
    # Ordenar los nodos de Chebyshev y sus valores
    sorted_indices = np.argsort(x_chebyshev)
    x_chebyshev_sorted = x_chebyshev[sorted_indices]
    y_chebyshev_sorted = y_chebyshev[sorted_indices]
    
    # Interpolaciones
    lineal_interpol = interp1d(x_chebyshev_sorted, y_chebyshev_sorted, kind='linear', bounds_error=False, fill_value="extrapolate")
    cubica_interpol = CubicSpline(x_chebyshev_sorted, y_chebyshev_sorted, extrapolate=True)
    
    # Calcular interpolaciones en los puntos intermedios
    z_lineal_interpol = lineal_interpol(puntos_intermedios)
    z_cubica_interpol = cubica_interpol(puntos_intermedios)
    
    # Calcular el error absoluto
    error_abs_lineal = np.abs(z_inter - z_lineal_interpol)
    error_abs_cubica = np.abs(z_inter - z_cubica_interpol)
    
    # Calcular el error relativo
    error_rel_lineal = error_abs_lineal / np.abs(z_inter)
    error_rel_cubica = error_abs_cubica / np.abs(z_inter)
    
    # Almacenar el error relativo mediano
    errores_lineal.append(np.median(error_rel_lineal))
    errores_cubica.append(np.median(error_rel_cubica))

# Graficar los errores relativos medianos
plt.figure(figsize=(10, 6))
plt.plot(nodos, errores_lineal, label='Interpolación Lineal', marker='o', linestyle='-')
plt.plot(nodos, errores_cubica, label='Interpolación Cúbica', marker='o', linestyle='--')
plt.xlabel('Número de Nodos')
plt.ylabel('Error Relativo Mediano')
plt.title('Error Relativo Mediano vs Número de Nodos (Nodos de Chebyshev)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()