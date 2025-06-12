# Métodos Numéricos y Optimización – Segundo Semestre 2024

## Análisis experimental y comparativo de Métodos Numéricos para la Interpolación y Trilateración de posiciones

Este repositorio contiene los códigos, datos y figuras empleados para el informe técnico que acompaña este análisis.  
En el informe (máx. 20 páginas) se analizan los siguientes experimentos numéricos:

1. **Efecto del número y la posición de los puntos de interpolación**  
   a. Función 1 – $f_1(x)= -0.4\,\tanh(50x)+0.6$ en $x\in[-1,1]$  
   b. Función 2 – $f_2(x_1,x_2)$ (definida en el enunciado) en $(x_1,x_2)\in[-1,1]^2$  
   Distintos esquemas de interpolación (Lagrange, lineal y splines/CubicSpline) se comparan usando:
   • nodos **equiespaciados**  
   • nodos **no-equiespaciados** (polos de Chebyshev)  
   Se reporta la precisión (error absoluto/relativo) en función de la cantidad y la distribución de nodos.

2. **Trilateración de posiciones en 3D**  
   A partir de distancias exactas medidas por tres sensores estáticos se reconstruye la trayectoria de una partícula utilizando el **método de Newton–Raphson**.  
   Luego se genera una trayectoria suave mediante **splines cúbicos** y se compara contra la trayectoria real.

---

## 1. Estructura del repositorio

```
.
├── README.md                 ← este archivo
│
├── Interpolación_f1.ipynb     ← Notebook Experimento 1 – f₁ (1D)
├── Interpolación_f2.ipynb     ← Notebook Experimento 1 – f₂ (2D)
├── Trilateración_de_posiciones.ipynb ← Notebook Experimento 2 – trilateración 3D
│
├── mnyo_tp01_datasets/        ← 📂 datos crudos para trilateración
│   ├── sensor_positions.txt
│   ├── medidas.csv            ← distancias d₁,d₂,d₃(t)
│   ├── trajectory.txt / trayectoria.csv
│   └── …
│
├── trayectoria_interpolada.csv  ← trayectoria reconstruida (salida)
├── posiciones relativas.csv      ← posiciones reconstruidas (salida)
│
├── Análisis_experimental___MNyO_Ortega_y_Götz.pdf  ← informe PDF
└── .venv/ , .git/ …
```

> Las rutas se expresan relativas a la raíz del proyecto.

---

## 2. Requisitos

Python ≥ 3.9

Instalar dependencias en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El archivo `requirements.txt` (no versionado por defecto) debe incluir:

```
numpy
pandas
matplotlib
scipy
```

---

## 3. Cómo reproducir cada experimento

Todas las rutas que se indican se ejecutan desde el directorio `TP 1/TP1 MÉTODOS`.
Todos los comandos se ejecutan desde la **raíz del proyecto**.

### 3.1 Interpolación – Función $f_1(x)$

| Notebook | Nodos | Descripción | Salidas (principales) |
|----------|-------|-------------|-----------------------|
| `Interpolación_f1.ipynb` | equiespaciados / Chebyshev (seleccionables) | Interpolación con Lagrange, lineal y spline; análisis del error vs número de nodos | Gráficos de función, error relativo y evolución del error. |

Los gráficos se guardan automáticamente en el mismo directorio y se muestran por pantalla.

### 3.2 Interpolación – Función $f_2(x_1,x_2)$

| Notebook | Nodos | Métodos | Salidas |
|----------|-------|---------|---------|
| `Interpolación_f2.ipynb` | equiespaciados / Chebyshev (seleccionables) | Interpolación lineal y cúbica vía `griddata` | Superficies 3D, mapas de calor y evolución del error. |

> Nota: la adaptación Chebyshev se implementa generando la malla con la transformada de coseno; para la versión 2D ambas coordenadas comparten la misma regla.

### 3.3 Trilateración 3D

Abrir y ejecutar todas las celdas de:

```bash
jupyter notebook Trilateración_de_posiciones.ipynb
```

El notebook realiza los siguientes pasos:

1. Carga `medidas.csv` con las distancias $d_1,d_2,d_3(t)$.
2. Resuelve las ecuaciones de trilateración aplicando **Newton–Raphson**.
3. Almacena las posiciones reconstruidas en `posiciones relativas.csv` y grafica la convergencia por iteración.
4. Ajusta **splines cúbicos** para suavizar la trayectoria y la compara con el ground-truth.
5. Guarda `trayectoria_interpolada.csv` y genera las figuras `posiciones.png`, `trayectoria_real_interpolada.png` y `evolucion_error.png`.

---

## 4. Resultados y figuras

Todos los gráficos y archivos CSV que se generan se ubican junto a cada script para facilitar la referencia desde el informe.  
Las figuras listadas en el informe deben copiarse a la carpeta `figuras/` (no incluida en el repositorio) para mantener el documento limpio.

---

## 5. Contacto

*Belén Götz – Francisco Ortega*  
Métodos Numéricos y Optimización – UDESA 2024 