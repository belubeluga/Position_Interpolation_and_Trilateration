import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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
    perf = [] 
    for i in range(max_iter):
        
        #Itero
        F = equations(x, d1, d2, d3) ## Calculo F(x)

        #Evaluo X en el Jacobiano
        J = jacobian(x)
        
        # Resolver J * delta_x = -F(x) (esto sale de Xn+1 = Xn - J^-1 * F(x))
        delta_x = np.linalg.solve(J, -F)
        
        # Actualizar x (paso iterativo). Xn+1 = Xn + delta_x
        x = x + delta_x
        perf.append((i,np.linalg.norm(delta_x)))
        # Verificar la convergencia. Deberia tender a cero
        if np.linalg.norm(delta_x) < tol:
            return x,perf
    
    raise RuntimeError("El método de Newton-Raphson no convergió")

#Voy a tirar sobre mis medidas para obtener una posicion(t)
medidas = pd.read_csv(r'M-TODOS-frantrabajando3008\mnyo_tp01_datasets\medidas.csv',sep=';')

t = np.array(medidas.iloc[:,0])
posiciones_x = []
posiciones_y = []
posiciones_z = []
rounds=[]

for i in range(len(medidas)):
    #Tomo mis valores de las distancias
    d1 = float(medidas.iloc[i,1])
    d2 = float(medidas.iloc[i,2])
    d3 = float(medidas.iloc[i,3])

    #Llamo a la funcion y devuelvo el vector de posiciones
    (x,y,z),perf = newton_raphson(d1, d2, d3)
    rounds.append(perf)
    #Registro la posicion en cada eje
    posiciones_x.append(round(x,4))
    posiciones_y.append(round(y,4))
    posiciones_z.append(round(z,4))

data = {
    't(s)': t,
    'X(m)': posiciones_x,
    'Y(m)': posiciones_y,
    'Z(M)': posiciones_z
}
df = pd.DataFrame(data)
# Guardar el DataFrame en un archivo CSV
df.to_csv('posicionesRelativas.csv', index=False)


# Preparar los datos para graficar
for iteration_data in rounds:
    if(rounds.index(iteration_data)==19):
        break
    iterations = [item[0] for item in iteration_data]
    delta_x = [item[1] for item in iteration_data]
    
    plt.plot(iterations, delta_x, marker='o', alpha = 0.8,label=f'Simulacion {rounds.index(iteration_data)}')

# Personalizar el gráfico
plt.xlabel('Iteración')
plt.ylabel('Delta x')

plt.xticks(ticks=range(len(iterations)), labels=iterations)
plt.title('Convergencia de Delta x en Iteraciones')
plt.legend()
#plt.yscale('log')  # Usa escala logarítmica si los valores varían en órdenes de magnitud
plt.grid(True)
# Guardar el gráfico en formato PDF
plt.savefig('Convergencia.png', format='png')

# Mostrar el gráfico
plt.show()








def spline_cubicos(t, posiciones_x, posiciones_y, posiciones_z):
    # Crear splines cúbicos para interpolar las posiciones en cada eje
    spline_x = CubicSpline(t, np.array(posiciones_x),bc_type="periodic")
    spline_y = CubicSpline(t, np.array(posiciones_y),bc_type="periodic")
    spline_z = CubicSpline(t, np.array(posiciones_z),bc_type="periodic")

    # Definir un nuevo conjunto de puntos de tiempo para la interpolación
    t_new = np.arange(min(t), max(t),0.01)
    t_new = np.append(t_new, 10)
    
    #Redondear los valores
    
    t_new = np.round(t_new,2)

    # Interpolar las posiciones usando los splines
    x_new = spline_x(t_new)
    y_new = spline_y(t_new)
    z_new = spline_z(t_new)

    return t_new, x_new, y_new, z_new

def plot_3d(t, xs, ys, zs, pos_x, pos_y, pos_z):
    # Graficar la trayectoria interpolada en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(xs, ys, zs, label='Trayectoria interpolada', color='b')
    ax.scatter(np.array(pos_x), np.array(pos_y), np.array(pos_z), color='r', label='Posiciones trilateradas')
    
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('Interpol.png', format='png')
    plt.legend()
    plt.show()



t_interpolado, tray_x_interpolado, tray_y_interpolado, tray_z_interpolado = spline_cubicos(t, posiciones_x, posiciones_y, posiciones_z)


plot_3d(t_interpolado, tray_x_interpolado, tray_y_interpolado, tray_z_interpolado, posiciones_x, posiciones_y, posiciones_z)

######
data = {
    't_interpolado': t_interpolado,
    'tray_x_interpolado': tray_x_interpolado,
    'tray_y_interpolado': tray_y_interpolado,
    'tray_z_interpolado': tray_z_interpolado
}
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv('trayectoria_interpolada.csv', index=False)


#Voy a leer la trayectoria y graficarla

trayectoria_df = pd.read_csv(r'M-TODOS-frantrabajando3008\mnyo_tp01_datasets\trayectoria.csv',sep=';')

t_reales = np.array(trayectoria_df.iloc[:,0])

tray_x_real = np.array(trayectoria_df.iloc[:,1])
tray_y_real = np.array(trayectoria_df.iloc[:,2])
tray_z_real = np.array(trayectoria_df.iloc[:,3])



eval_errores = []
for i in range(len(t_reales)):
    
    error = np.sqrt((tray_x_real[i] - tray_x_interpolado[i])**2 + (tray_y_real[i] - tray_y_interpolado[i])**2 + (tray_z_real[i] - tray_z_interpolado[i])**2)
    eval_errores.append((t_reales[i], error))
    
#Calcular la mediana del error 
errores_val = [error for t, error in eval_errores]
mediana_error = round(np.median(errores_val),5)
#Calcular el error maximo
max_error = round(max(errores_val),5)

# Graficar la trayectoria real y la interpolada en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(tray_x_interpolado, tray_y_interpolado, tray_z_interpolado, label='Trayectoria interpolada', color='b')
ax.plot(tray_x_real, tray_y_real, tray_z_real, label='Trayectoria real', color='r',linewidth=4,       # Ancho de la línea
         linestyle='--',
         alpha = 0.4)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('Interpol2.png', format='png')
plt.legend()
plt.show()

#Graficar la evolucion del error en el tiempo
t_eval, errores = zip(*eval_errores)
plt.plot(t_eval, errores, label='Error')
plt.xlabel('Tiempo(s)')
plt.ylabel('Error')
plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.savefig('Error.png', format='png')
plt.show()

