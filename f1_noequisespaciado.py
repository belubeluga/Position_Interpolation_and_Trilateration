import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy

# función 1 f1(x)
def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

def f_prime(x):
    return -0.4 * 50 * (1 - np.tanh(50 * x)**2)

# Derivada segunda: -0.4 * 50^2 * (-2 * tanh(50 * x)) * sech^2(50 * x)
def f_double_prime(x):
    sech_squared = (1 - np.tanh(50 * x)**2)

    return -0.4 * 50**2 * (-2 * np.tanh(50 * x)) * sech_squared

errores_langrange_dic = {}
errores_cubica_dic = {}
errores_lineal_dic = {}

# Armar distintos rangos no equisespaciados



min_langrange = 100000000
min_cubica = 100000000
min_lineal = 100000000

puntos_x = np.linspace(-1, 1, 1000)

for i in range(2,10):
    for j in range (1,13):
            
            a = np.linspace(-1, -0.2, i ,endpoint=True)
            b = np.linspace(-0.2, 0.2, j,endpoint=False)
            c = np.linspace(0.2, 1, i,endpoint=True)
            

            segmento = np.concatenate((a,b,c))
            
            segmento = np.unique(segmento)

            
            lagrangeInterpol1 = scipy.lagrange(segmento, f1(segmento))
            splineCubicaInterpol1 = scipy.CubicSpline(segmento, f1(segmento))
            splineLinealInterpol1 = scipy.interp1d(segmento, f1(segmento), kind='linear')

            errores_langrange = []
            errores_cubica = []
            errores_lineal = []

            puntos_x = np.linspace(min(segmento), max(segmento), 100, endpoint=True)
            # Medir el error
            for x in puntos_x:

                if x in segmento:
                    continue

                errores_langrange.append( abs(f1(x) - lagrangeInterpol1(x)) )
                errores_cubica.append( abs(f1(x) - splineCubicaInterpol1(x)) )
                errores_lineal.append( abs(f1(x) - splineLinealInterpol1(x)) )
                
                errores_langrange_mediana = np.mean(errores_langrange)
                errores_cubica_mediana = np.mean(errores_cubica)
                errores_lineal_mediana = np.mean(errores_lineal)

                errores_langrange_dic[(i,j)] = errores_langrange_mediana
                errores_cubica_dic[(i,j)] = errores_cubica_mediana
                errores_lineal_dic[(i,j)] = errores_lineal_mediana

                if  errores_langrange_mediana < min_langrange:
                    min_langrange = errores_langrange_mediana
                    lagrange_interpolacion = lagrangeInterpol1(puntos_x)

                if  errores_cubica_mediana < min_cubica:
                    min_cubica = errores_cubica_mediana
                    cubica_interpolacion = splineCubicaInterpol1(puntos_x)

                if  errores_lineal_mediana < min_lineal:
                    min_lineal = errores_lineal_mediana
                    lineal_interpolacion = splineLinealInterpol1(puntos_x)








#Imprimir los errores
print(f'El mejor error en lagrange fue {min(errores_langrange_dic.values())} con {min(errores_langrange_dic, key=errores_langrange_dic.get)} puntos')
print(f'El mejor error en cubica fue {min(errores_cubica_dic.values())} con {min(errores_cubica_dic, key=errores_cubica_dic.get)} puntos')
print(f'El mejor error en lineal fue {min(errores_lineal_dic.values())} con {min(errores_lineal_dic, key=errores_lineal_dic.get)} puntos')


# Graficar las funciones interpoladas
plt.figure(figsize=(10, 5))
plt.plot(puntos_x, f1(puntos_x), label='f1(x)', color='black')

plt.plot(puntos_x, lagrange_interpolacion, label='Polinomio de Lagrange', color='red',linestyle='--')
plt.plot(puntos_x, cubica_interpolacion , label='Spline cúbica', color='blue',linestyle='--')
plt.plot(puntos_x, lineal_interpolacion, label='Spline lineal', color='green',linestyle='--')

plt.xlabel('x')
plt.ylabel('f1(x)')
plt.ylim(0, 1.5)  # Ajustar el eje y
plt.title('Interpolación de f1(x)')
plt.legend()
plt.grid(True)
plt.show()




# Graficar el segmento
plt.figure(figsize=(10, 2))
plt.plot(segmento, np.zeros_like(segmento), 'o', label='Segmento')
plt.yticks([])  # Ocultar el eje y
plt.xlabel('Valores de segmento')
plt.title('Gráfico de segmento concatenado')
plt.legend()
plt.grid(True)
plt.show()


 