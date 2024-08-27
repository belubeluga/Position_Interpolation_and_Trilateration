import numpy as np
import matplotlib.pyplot as plt

# Definir la función f1(x)
def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

# Crear los valores de x
x = np.linspace(-1, 1, 400)

# Calcular los valores de f1(x)
y = f1(x)

# Graficar f1(x)
plt.plot(x, y)
plt.title('$f_1(x) = -0.4 \\tanh(50x) + 0.6$')
plt.xlabel('x')
plt.ylabel('$f_1(x)$')
plt.grid(True)
plt.show()