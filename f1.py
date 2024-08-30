import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy

# función 1 f1(x)
def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

""" para graficar f1(x) """
# plt.plot(x, y)
# plt.title('$f_1(x) = -0.4 \\tanh(50x) + 0.6$')
# plt.xlabel('x')
# plt.ylabel('$f_1(x)$')
# plt.grid(True)
# plt.show() 

""" EVALUANDO EN LOS PUNTOS QUE USAMOS PARA ARMAR EL LAGRANGE """
""" ( LAGRANGE EQUIESPACIADO ) """

erroresEnDataset = [] #errores propios de trabajar con floats

for i in range(1, 50):
    break
    x1 = np.linspace(-1, 1, i*5) # creamos los valores de x con los que vamos a interpolar
    y1 = f1(x1) #nodos/ ground thruths / DATASET los valores de f1(x)

    lagrangeInterpol1 = scipy.lagrange(x1, y1)

    #análisis del error

    eLagrange = [] #lista con el error en cada aprox

    for j in range(0, len(x1)):
        eLagrange.append( abs(y1[j] - lagrangeInterpol1(x1[j])) ) 

    error = np.median(eLagrange)

    erroresEnDataset.append((int(error),int(i*5)))

"""
print("EVALUANDO EN LOS PUNTOS DEL DATASET: ")
print("error aproximando con lagrange: ", np.median(eLagrange))
print("error aproximando con CubicSpline: ", np.median(eCS)) 
"""

#menor_error = np.min(erroresEnDataset)
#print("El menor número es:", menor_error) #2.220446049250313e-16

#print(np.array(errores))


""" EVALUACIÓN EN PUNTOS INTERMEDIOS """

errores_langrange_dic = {}
errores_cubica_dic = {}
errores_lineal_dic = {}
puntos_x = np.linspace(-1, 1, 1000)

min_langrange = 100
min_cubica = 100
min_lineal = 100

for i in range(5,50):
    ##Sub divido mi intervalo de -1 a 1 en 10*i puntos
    x2 = np.linspace(-1, 1, i)
    #Calculo la imagen
    y2 = f1(x2)
    
    errores_langrange = []
    errores_cubica = []
    errores_lineal = []

    polinomioDeLangrange = scipy.lagrange(x2, y2)
    interpol_lineal = scipy.interp1d(x2, y2, kind='linear')
    interpol_cubica = scipy.interp1d(x2, y2, kind='cubic')

    
    puntosIntermedios = np.linspace(-1, 1, 2*i, endpoint=False)
    
    for x in puntosIntermedios:
        if x in x2:
            continue

        errores_langrange.append( abs(f1(x) - polinomioDeLangrange(x)) )
        errores_cubica.append( abs(f1(x) - interpol_cubica(x)) )
        errores_lineal.append( abs(f1(x) - interpol_lineal(x)) )
       
    errores_langrange_mediana = np.median(errores_langrange)
    errores_cubica_mediana = np.median(errores_cubica)
    errores_lineal_mediana = np.median(errores_lineal)

    errores_langrange_dic[i] = errores_langrange_mediana
    errores_cubica_dic[i] = errores_cubica_mediana
    errores_lineal_dic[i] = errores_lineal_mediana

    if  errores_langrange_mediana < min_langrange:
        min_langrange = errores_langrange_mediana
        lagrange_interpolacion = polinomioDeLangrange(puntos_x)

    if  errores_cubica_mediana < min_cubica:
        min_cubica = errores_cubica_mediana
        cubica_interpolacion = interpol_cubica(puntos_x)

    if  errores_lineal_mediana < min_lineal:
        min_lineal = errores_lineal_mediana
        lineal_interpolacion = interpol_lineal(puntos_x)
    
    
        



print(f'El mejor error en lagrange fue {min(errores_langrange_dic.values())} con {min(errores_langrange_dic, key=errores_langrange_dic.get)} puntos')
print(f'El mejor error en cubica fue {min(errores_cubica_dic.values())} con {min(errores_cubica_dic, key=errores_cubica_dic.get)} puntos')
print(f'El mejor error en lineal fue {min(errores_lineal_dic.values())} con {min(errores_lineal_dic, key=errores_lineal_dic.get)} puntos')

# Ploteo de las funciones interpoladas

plt.figure(figsize=(10, 6))
plt.plot(puntos_x, f1(puntos_x), label="Función original", color='black')
plt.plot(puntos_x, lagrange_interpolacion, label="Interpolación Lagrange", linestyle='--')
plt.plot(puntos_x, cubica_interpolacion, label="Interpolación Cúbica", linestyle='--')
plt.plot(puntos_x, lineal_interpolacion, label="Interpolación Lineal", linestyle='--')
plt.legend()
plt.title("Interpolaciones")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)

plt.xlim([-1, 1])
plt.ylim([0, 2])

plt.show()



#print(errores_langrange_dic)
#print(errores_cubica_dic)
#print(errores_lineal_dic)
"""
media_de_error cantidad_de_puntos_equisespaciados
0.4 2
0.4 4
0.39999999999999997 6
0.3999999999999997 8
0.3999999999999996 10
0.39999999999999986 12
0.4000000000000007 14
0.39999999999999997 16
0.40000000000000024 18
0.40000000000000063 20
0.4000000000000027 22
0.3999999999999995 24
0.39999999999999963 26
0.3999999999999995 28
0.39999999999999847 30
0.7693747226407456 32
0.7999999975579353 34
0.7976407062002694 36
0.7997945460752511 38
"""    






""" primer observacion: ELECCIÓN DE CANTIDAD DE DATOS

( LAGRANGE EQUIESPACIADO )

RUNGE: mientras mas iteraciones, menos precision con lagrange
(de 100 a 1000)
[1.13797860e-15 1.01432751e-11 1.35347009e-08 6.36585596e-05
 4.08150754e-02 1.53964245e+02 1.20567439e+06 1.83115839e+09
 8.21370479e+12] 
 
evaluar LINESPACE PARA LAGRANGE
-> al ser equiespaciado el linespace va a ser parejo en todo el dominio => va a depender de la cantidad de puntos """


""" evaluar LINESPACE PARA LAGRANGE
-> 
"""


"""
------> comparaciones a hacer:
lagrangeInterpol = scipy.lagrange(x, y)
csInterpol = scipy.CubicSpline(x, y)

eLagrange = np.zeros(len(x))
eCS = np.zeros(len(x))

for i in range(0, len(x)):
    eLagrange[i] = abs(y[i] - lagrangeInterpol(x[i]))
    eCS[i] = abs(y[i] - csInterpol(x[i]))
"""