import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy
from scipy.optimize import minimize_scalar

def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

res_min = minimize_scalar(f1, bounds=(-1, 1), method='bounded')
res_max = minimize_scalar(lambda x: -f1(x), bounds=(-1, 1), method='bounded')

min_y = res_min.fun
max_y = -res_max.fun

min_x = res_min.x
max_x = res_max.x

print(f"Min ({min_x};{min_y}), Max ({max_x};{max_y})")

# Chebyshev: https://en.wikipedia.org/wiki/Chebyshev_nodes (Pecado que ayer me olvide de su existencia)
def chebyshev_nodes(a, b, n):
    return 0.5*(b - a) * np.cos((2*np.arange(1, n+1) - 1) / (2*n) * np.pi) + 0.5*(b + a)

min_error = float('inf')
min_inter = None
min_err_seg = None

for n in range(5, 100, 5):
    segmento = chebyshev_nodes(-1, 1, n)
    lagrange = scipy.lagrange(segmento, f1(segmento))

    puntos_x = np.linspace(-1, 1, 1000)
    errores_langrange = np.abs(f1(puntos_x) - lagrange(puntos_x))
    error_mediana = np.mean(errores_langrange)

    if error_mediana < min_error:
        min_error = error_mediana
        min_inter = lagrange(puntos_x)
        min_err_seg = segmento

print(f'Lagrange: {min_error} - {len(min_err_seg)}')

# Plot - Para que existis...
plt.figure(figsize=(10, 5))
plt.plot(puntos_x, f1(puntos_x), label='f1(x)?', color='black')
plt.plot(puntos_x, min_inter, label='La-narange (Italiano)', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('No es f1(x)')
plt.ylim(0.1, 1.2)
plt.title('Si esto funciona, ya puedo morir en paz')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 2))
plt.plot(min_err_seg, np.zeros_like(min_err_seg), 'o', label='Distribucion de los astros')
plt.yticks([])
plt.xlabel('????')
plt.title('Tirados en el suelo')
plt.legend()
plt.grid(True)
plt.show()