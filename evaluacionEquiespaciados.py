import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, interp1d, CubicSpline

""" ANALISIS DE METODOS EN INTERVALOR EQUIESPACIADOS """
# función f1
def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

# Puntos equiespaciados
n_points = 2
x_eq = np.linspace(-1, 1, n_points)
y_eq = f1(x_eq)

# Puntos de evaluación INTERMEDIOS (que no coincidan con los NODOS)
puntosIntermedios = np.linspace(-1, 1, 500)
puntosIntermedios = np.setdiff1d(puntosIntermedios, x_eq)  # Excluir los nodos de interpolación
y_intermedios = f1(puntosIntermedios)

# Interpolaciones
lagrangeInterpol = lagrange(x_eq, y_eq)
linealInterpol = interp1d(x_eq, y_eq, kind='linear')
splineInterpol = CubicSpline(x_eq, y_eq)

# para evaluar errores relativos
eLineal = []  
eLagrange = []
eSpline = []

for x in puntosIntermedios:
    eLagrange.append(abs(f1(x) - lagrangeInterpol(x)) / f1(x))
    eLineal.append(abs(f1(x) - linealInterpol(x)) / f1(x))
    eSpline.append(abs(f1(x) - splineInterpol(x)) / f1(x))

# Calculamos la mediana de los errores relativos de la interpolación
errorLagrange = np.median(eLagrange)
errorLineal = np.median(eLineal)
errorSpline = np.median(eSpline)
print(f"Cantidad de nodos usados para interpolar: ", n_points)
print(f"Error mediano de interpolación Lagrange:", errorLagrange)
print(f"Error mediano de interpolación Lineal:", errorLineal)
print(f"Error mediano de interpolación Spline:", errorSpline)

# Puntos nuevos para la evaluación
y_linear_intermedios = linealInterpol(puntosIntermedios)
y_lagrange_intermedios = lagrangeInterpol(puntosIntermedios)
y_spline_intermedios = splineInterpol(puntosIntermedios)

# Mostrar los resultados en un gráfico
plt.figure(figsize=(12, 6))

# Puntos equiespaciados
plt.plot(x_eq, y_eq, 'o', label='Puntos equiespaciados')
plt.plot(puntosIntermedios, y_linear_intermedios, ':', label='Interpolación Lineal')
plt.plot(puntosIntermedios, y_lagrange_intermedios, '--', label='Interpolación Lagrange')
plt.plot(puntosIntermedios, y_spline_intermedios, '-.', label='Interpolación Spline')
plt.plot(puntosIntermedios, y_intermedios, '-', label='Función Original')
plt.legend()
plt.title('Interpolación Lineal, Lagrange y Spline con Puntos Equiespaciados')
plt.xlabel('x')
plt.ylabel('$f_1(x)$')
plt.grid(True)

print("Puntos de interpolación equiespaciados:", x_eq)

# Ajustar la escala del gráfico
plt.ylim(min(y_intermedios) - 1, max(y_intermedios) + 1)
plt.xlim(-1, 1)

plt.tight_layout()
plt.show()



"""
COMPARACION ERRORES:

Cantidad de nodos usados para interpolar:  2
Error mediano de interpolación Lagrange: 0.3246492949657289
Error mediano de interpolación Lineal: 0.3246492949657289
Error mediano de interpolación Spline: 0.3246492949657289
- error es mayor en todos pero es parejo

Cantidad de nodos usados para interpolar:  5
Error mediano de interpolación Lagrange: 0.1920038388918091
Error mediano de interpolación Lineal: 0.002404809619238446
Error mediano de interpolación Spline: 0.1920038388918098
- mejora muy rapidamente lineal (todos se achican)

Cantidad de nodos usados para interpolar:  10
Error mediano de interpolación Lagrange: 0.31406663706823446
Error mediano de interpolación Lineal: 1.0547118733938995e-15
Error mediano de interpolación Spline: 0.027711353961269176
- lineal mejora mucho, spline mejora bastante, lagrange empieza a empeorar

Cantidad de nodos usados para interpolar:  20
Error mediano de interpolación Lagrange: 0.4881649714402162
Error mediano de interpolación Lineal: 0.0
Error mediano de interpolación Spline: 0.0005667519379985027
- lineal funciona muy bien, spline tambien, lagrange sigue empeorando

Cantidad de nodos usados para interpolar:  40
Error mediano de interpolación Lagrange: 1.6242023826716987
Error mediano de interpolación Lineal: 0.0
Error mediano de interpolación Spline: 4.78766336783898e-07
- lagrange subio un monton, lineal sigue bajo, spline tmb 



Interpolación Lineal:
Funciona muy bien con un número creciente de nodos.
El error mediano es extremadamente bajo o cero a partir de 10 nodos.
Recomendado para la mayoría de los casos debido a su simplicidad y precisión.

Interpolación Spline:
Mejora significativamente con más nodos.
El error mediano es muy bajo a partir de 10 nodos.
Recomendado cuando se necesita una interpolación suave y precisa.

Interpolación de Lagrange:
Empieza a empeorar con un número creciente de nodos.
No recomendado para un gran número de nodos debido a la inestabilidad y el aumento del error.

La interpolación lineal es la mejor opción debido a su simplicidad y precisión. 
La interpolación spline es una excelente alternativa cuando se necesita una interpolación más suave. 
La interpolación de Lagrange no es recomendada para un gran número de nodos debido a su inestabilidad y aumento del error.

"""