import numpy as np
import pandas as pd 

#posicionSensores = [(10,0,0),(0,10,0),(0,0,10)]

def equations(vars, d1, d2, d3):
    x, y, z = vars
    eq1 = (x - 10)**2 + y**2 + z**2 - d1**2
    eq2 = (y - 10)**2 + x**2 + z**2 - d2**2
    eq3 = (z - 10)**2 + y**2 + x**2 - d3**2
    return np.array([eq1, eq2, eq3])

def jacobian(vars):
    x, y, z = vars
    J = np.array([
        [2 * (x - 10), 2 * y, 2 * z],
        [2 * x, 2 * (y - 10), 2 * z],
        [2 * x, 2 * y, 2 * (z - 10)]
    ])
    return J

# Método de Newton-Raphson
def newton_raphson( d1, d2, d3, tol=1e-6, max_iter=100):

    x = (0, 0, 0)  ## Armo mi punto inicial X0 

    for i in range(max_iter):
        #Itero
        F = equations(x, d1, d2, d3) ## Calculo F(x)

        #Evaluo X en el Jacobiano
        J = jacobian(x)
        
        # Resolver J * delta_x = -F(x) (esto sale de Xn+1 = Xn - J^-1 * F(x))
        delta_x = np.linalg.solve(J, -F)
        
        # Actualizar x (paso iterativo). Xn+1 = Xn + delta_x
        x = x + delta_x
        
        # Verificar la convergencia. Deberia tender a cero
        if np.linalg.norm(delta_x) < tol:
            return x
    
    raise RuntimeError("El método de Newton-Raphson no convergió")

#Voy a tirar sobre mis medidas para obtener una posicion(t)
medidas = pd.read_csv('mnyo_tp01_datasets\medidas.csv',sep=';')

posiciones = []

for i in range(len(medidas)):
    #Tomo mis valores de las distancias
    d1 = float(medidas.iloc[i,1])
    d2 = float(medidas.iloc[i,2])
    d3 = float(medidas.iloc[i,3])

    #Llamo a la funcion y devuelvo el vector de posiciones
    x,y,z = newton_raphson(d1, d2, d3)
    #Registro la posicion
    posiciones.append((x,y,z))

print(posiciones)




