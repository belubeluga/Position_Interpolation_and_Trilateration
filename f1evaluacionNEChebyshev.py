import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, interp1d, CubicSpline

# Función original
def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

# Generar puntos de Chebyshev
n_points = 100
i = np.arange(n_points)
x_chebyshev = np.cos((2 * i + 1) / (2 * n_points) * np.pi)
y_chebyshev = f1(x_chebyshev)

# Ordenar los puntos de Chebyshev
sorted_indices = np.argsort(x_chebyshev)
x_chebyshev = x_chebyshev[sorted_indices]
y_chebyshev = y_chebyshev[sorted_indices]

# Puntos de evaluación INTERMEDIOS (que no coincidan con los NODOS)
puntosIntermedios = np.linspace(-1, 1, 500)
puntosIntermedios = np.setdiff1d(puntosIntermedios, x_chebyshev)  # Excluir los nodos de interpolación

# Filtrar puntos de evaluación para que estén dentro del rango de los nodos de Chebyshev
min_x = x_chebyshev[0]
max_x = x_chebyshev[-1]
puntosIntermedios = puntosIntermedios[(puntosIntermedios > min_x) & (puntosIntermedios < max_x)]
y_intermedios = f1(puntosIntermedios)

# Puntos nuevos para la evaluación
lagrangeInterpol = lagrange(x_chebyshev, y_chebyshev)
linealInterpol = interp1d(x_chebyshev, y_chebyshev, kind='linear')
splineInterpol = CubicSpline(x_chebyshev, y_chebyshev)

# Evaluar las interpolaciones en los puntos intermedios
y_lagrange_intermedios = lagrangeInterpol(puntosIntermedios)
y_linear_intermedios = linealInterpol(puntosIntermedios)
y_spline_intermedios = splineInterpol(puntosIntermedios)


# Calcular errores relativos
eLagrange = []
eLineal = []
eSpline = []

for x in puntosIntermedios:
    eLagrange.append(abs(f1(x) - lagrangeInterpol(x)) / f1(x))
    eLineal.append(abs(f1(x) - linealInterpol(x)) / f1(x))
    eSpline.append(abs(f1(x) - splineInterpol(x)) / f1(x))

# Calcular la mediana de los errores relativos de la interpolación
errorLagrange = np.median(eLagrange)
errorLineal = np.median(eLineal)
errorSpline = np.median(eSpline)
print(f"Cantidad de nodos usados para interpolar: {n_points}")
print(f"Error mediano de interpolación Lagrange: {errorLagrange}")
print(f"Error mediano de interpolación Lineal: {errorLineal}")
print(f"Error mediano de interpolación Spline: {errorSpline}")

# Mostrar los resultados en un gráfico
plt.figure(figsize=(12, 6))

# Puntos de Chebyshev
plt.plot(x_chebyshev, y_chebyshev, 'o', label='Puntos de Chebyshev')

# Interpolaciones y función original evaluadas en puntos intermedios
plt.plot(puntosIntermedios, y_linear_intermedios, ':', label='Interpolación Lineal')
plt.plot(puntosIntermedios, y_lagrange_intermedios, '--', label='Interpolación Lagrange')
plt.plot(puntosIntermedios, y_spline_intermedios, '-.', label='Interpolación Spline')
plt.plot(puntosIntermedios, y_intermedios, '-', label='Función Original')
plt.legend()
plt.title('Interpolación Lineal, Lagrange y Spline con Puntos de Chebyshev')
plt.xlabel('x')
plt.ylabel('$f_1(x)$')
plt.grid(True)

# Ajustar la escala del gráfico
plt.ylim(min(y_intermedios) - 1, max(y_intermedios) + 1)
plt.xlim(-1, 1)

plt.tight_layout()
plt.show()

"""

Cantidad de nodos usados para interpolar: 2
Error mediano de interpolación Lagrange: 0.32064488304487426
Error mediano de interpolación Lineal: 0.3206448830448744
Error mediano de interpolación Spline: 0.3206448830448744

Cantidad de nodos usados para interpolar: 5
Error mediano de interpolación Lagrange: 0.15243338319375216
Error mediano de interpolación Lineal: 0.1239863923210272
Error mediano de interpolación Spline: 0.15243338319375163

Cantidad de nodos usados para interpolar: 10
Error mediano de interpolación Lagrange: 0.10550991381668151
Error mediano de interpolación Lineal: 0.0
Error mediano de interpolación Spline: 0.027955773456514

Cantidad de nodos usados para interpolar: 20
Error mediano de interpolación Lagrange: 0.05837265707683721
Error mediano de interpolación Lineal: 0.0
Error mediano de interpolación Spline: 0.0036900312004303636

Cantidad de nodos usados para interpolar: 40
Error mediano de interpolación Lagrange: 0.04601748082792184
Error mediano de interpolación Lineal: 0.0
Error mediano de interpolación Spline: 2.9448112304819585e-05

Cantidad de nodos usados para interpolar: 100
Error mediano de interpolación Lagrange: 1912639347.3261356
Error mediano de interpolación Lineal: 0.0
Error mediano de interpolación Spline: 8.448020061280204e-12

"""