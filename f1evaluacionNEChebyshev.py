import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, interp1d, CubicSpline

# Función original
def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

# Generar puntos de Chebyshev
n_points = 20  # Usaremos 20 puntos no equidistantes (Chebyshev)
i = np.arange(n_points)
x_chebyshev = np.cos((2 * i + 1) / (2 * n_points) * np.pi)
y_chebyshev = f1(x_chebyshev)

# Ordenar los puntos de Chebyshev
sorted_indices = np.argsort(x_chebyshev)
x_chebyshev = x_chebyshev[sorted_indices]
y_chebyshev = y_chebyshev[sorted_indices]

# Puntos de evaluación intermedios (que no coincidan con los nodos)
puntosIntermedios = np.linspace(-1, 1, 500)
puntosIntermedios = np.setdiff1d(puntosIntermedios, x_chebyshev)  # Excluir los nodos de interpolación

# Filtrar puntos intermedios para que estén dentro del rango de los nodos de Chebyshev
min_x = x_chebyshev[0]
max_x = x_chebyshev[-1]
puntosIntermedios = puntosIntermedios[(puntosIntermedios > min_x) & (puntosIntermedios < max_x)]
y_intermedios = f1(puntosIntermedios)

# Interpolaciones
lagrangeInterpol = lagrange(x_chebyshev, y_chebyshev)
linealInterpol = interp1d(x_chebyshev, y_chebyshev, kind='linear')
splineInterpol = CubicSpline(x_chebyshev, y_chebyshev)

# Evaluar las interpolaciones en los puntos intermedios
y_lagrange_intermedios = lagrangeInterpol(puntosIntermedios)
y_lineal_intermedios = linealInterpol(puntosIntermedios)
y_spline_intermedios = splineInterpol(puntosIntermedios)

# Calcular errores relativos
error_rel_lineal = np.abs(y_intermedios - y_lineal_intermedios) / np.abs(y_intermedios)
error_rel_lagrange = np.abs(y_intermedios - y_lagrange_intermedios) / np.abs(y_intermedios)
error_rel_spline = np.abs(y_intermedios - y_spline_intermedios) / np.abs(y_intermedios)


# Gráfico de la función original y las interpolaciones
plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(puntosIntermedios, y_intermedios, '-', color = 'purple', label='Función Original')
plt.scatter(x_chebyshev, y_chebyshev, label='Nodos', zorder=5)
plt.plot(puntosIntermedios, y_lineal_intermedios, '--', label='Interpolación Lineal')
plt.plot(puntosIntermedios, y_lagrange_intermedios, '-.', label='Interpolación Lagrange')
plt.plot(puntosIntermedios, y_spline_intermedios, ':', label='Interpolación Spline')

plt.title('Función Original e Interpolaciones')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Gráficos de los errores relativos
plt.figure(figsize=(10, 8))

# Gráfico del error relativo para la interpolación lineal
plt.subplot(3, 1, 1)
plt.plot(puntosIntermedios, error_rel_lineal, color='orange', label='Error Relativo Lineal')
plt.title('Error Relativo - Interpolación Lineal')
plt.xlabel('x')
plt.ylabel('Error Relativo')
plt.grid(True)
plt.legend()

# Gráfico del error relativo para la interpolación spline cúbica
plt.subplot(3, 1, 2)
plt.plot(puntosIntermedios, error_rel_spline, color='red', label='Error Relativo Spline Cúbica')
plt.title('Error Relativo - Interpolación Spline Cúbica')
plt.xlabel('x')
plt.ylabel('Error Relativo')
plt.grid(True)
plt.legend()

# Gráfico del error relativo para la interpolación de Lagrange
plt.subplot(3, 1, 3)
plt.plot(puntosIntermedios, error_rel_lagrange, color='green', label='Error Relativo Lagrange')
plt.title('Error Relativo - Interpolación de Lagrange')
plt.xlabel('x')
plt.ylabel('Error Relativo')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
