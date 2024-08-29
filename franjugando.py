import numpy as np
import matplotlib.pyplot as plt

originales = np.arange(-1, 1, 0.01)
analiticos = []

# Función original: -0.4 * tanh(50 * x) + 0.6
def f(x):
    return -0.4 * np.tanh(50 * x) + 0.6

# Derivada primera: -0.4 * 50 * sech^2(50 * x)
def f_prime(x):
    return -0.4 * 50 * (1 - np.tanh(50 * x)**2)

# Derivada segunda: -0.4 * 50^2 * (-2 * tanh(50 * x)) * sech^2(50 * x)
def f_double_prime(x):
    sech_squared = (1 - np.tanh(50 * x)**2)
    return -0.4 * 50**2 * (-2 * np.tanh(50 * x)) * sech_squared


for celda in originales:
    eval = f_double_prime(celda)
    if abs(eval) > 0.01:
        cantidadDeCeldasNuevas = round(abs(eval) / 0.01)
        rango = np.linspace(celda, celda+0.01, cantidadDeCeldasNuevas)
        for elem in rango:
            analiticos.append(round(elem, 4))
        continue
    else:
        analiticos.append(round(celda, 4))
    #analiticos.append(np.linspace(celda, celda+0,1, 1))

# Eliminar duplicados y ordenar la lista
analiticos = sorted(list(set(analiticos)))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 1))
plt.scatter(analiticos, [0]*len(analiticos), color='blue')
plt.xlim(-1, 1)


# Usar una escala logarítmica en el eje x
plt.yticks([])
plt.xlabel('Valores únicos')
plt.title('Valores únicos en una recta de -1 a 1')
plt.grid(True)
plt.show()
    

