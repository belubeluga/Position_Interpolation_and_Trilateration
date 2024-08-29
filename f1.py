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
erroresEnDataset = [] #errores propios de trabajar con floats

for i in range(1, 50):

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
errores = []

for i in range(1, 20):
    x2 = np.linspace(-1, 1, i*2)
    y2 = f1(x1)
    
    eLagrange2 = []

    lagrangeInterpol2 = scipy.lagrange(x2, y2)
    
    puntosIntermedios = np.linspace(-1, 1, i*10)
    for x in puntosIntermedios:
        if x in x2:
            continue
        eLagrange2.append( abs(f1(x) - lagrangeInterpol2(x)) )

    error2 = np.median(eLagrange2)
    print(error2, i*2)   
     
"""
0.4
0.4
0.39999999999999997
0.3999999999999997
0.3999999999999996
0.39999999999999986
0.4000000000000007
0.39999999999999997
0.40000000000000024
0.40000000000000063
0.4000000000000027
0.3999999999999995
0.39999999999999963
0.3999999999999995
0.39999999999999847
0.7693747226407456
0.7999999975579353
0.7976407062002694
0.7997945460752511
"""    






""" primer observacion: ELECCIÓN DE CANTIDAD DE DATOS/ LINESPACE PARA LAGRANGE
RUNGE: mientras mas iteraciones, menos precision con lagrange
(de 100 a 1000)
[1.13797860e-15 1.01432751e-11 1.35347009e-08 6.36585596e-05
 4.08150754e-02 1.53964245e+02 1.20567439e+06 1.83115839e+09
 8.21370479e+12] """





"""
lagrangeInterpol = scipy.lagrange(x, y)
csInterpol = scipy.CubicSpline(x, y)

eLagrange = np.zeros(len(x))
eCS = np.zeros(len(x))

for i in range(0, len(x)):
    eLagrange[i] = abs(y[i] - lagrangeInterpol(x[i]))
    eCS[i] = abs(y[i] - csInterpol(x[i]))
"""