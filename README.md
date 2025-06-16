# Numerical Methods and Optimization – Second Semester 2024

## Experimental and Comparative Analysis of Numerical Methods for Interpolation and Position Trilateration

This repository contains the code, data, and figures used in the technical report accompanying this analysis.  
The report (max. 20 pages) presents the following numerical experiments:

---

### 1. Effect of the Number and Position of Interpolation Points

- **Function 1:**  
  \( f_1(x) = -0.4\,\tanh(50x) + 0.6 \), for \( x \in [-1, 1] \)

- **Function 2:**  
  \( f_2(x_1, x_2) \), defined in the assignment, for \( (x_1, x_2) \in [-1, 1]^2 \)

**Interpolation schemes compared:**
- Lagrange
- Linear
- Cubic splines (`CubicSpline`)

**Node types used:**
- **Equally spaced nodes**
- **Non-uniform nodes** (Chebyshev poles)

Accuracy (absolute and relative error) is reported as a function of the number and distribution of nodes.

---

### 2. 3D Position Trilateration

Using exact distances measured by three static sensors, the trajectory of a particle is reconstructed via the **Newton–Raphson method**.  
A smooth trajectory is then generated using **cubic splines** and compared to the real (ground-truth) trajectory.

---

## 📁 1. Repository Structure

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

> All paths are relative to the root of the project.

---

## ✅ 2. Requirements

```
numpy
pandas
matplotlib
scipy
```

---

## 3. How to reproduce each experiment
All paths mentioned here are relative to the `TP 1/TP1 MÉTODOS` directory.
All commands should be run from the **project root**.

### 3.1 Interpolation – Function $f_1(x)$

| Notebook | Nodes | Description | Main Outputs |
|----------|-------|-------------|-----------------------|
| `Interpolación_f1.ipynb` | Equally spaced / Chebyshev (selectable) | Interpolation using Lagrange, linear, and spline methods; error analysis by node count | Function plots, relative error, error evolution |

Plots are automatically saved in the same directory and also displayed during execution.

### 3.2 Interpolation – Function $f_2(x_1,x_2)$

| Notebook | Nodes | Methods | Main Outputs |
|----------|-------|---------|---------|
| `Interpolación_f2.ipynb` | Equally spaced / Chebyshev (selectable) | Linear and cubic interpolation via `griddata` | 3D surfaces, heatmaps, error plots, error evolution |

> Note: The Chebyshev adaptation is implemented using a cosine transform; in the 2D case, both axes follow the same rule..

### 3.3 Trilateration 3D

To run the experiment, open and execute all cells in:

```bash
jupyter notebook Trilateración_de_posiciones.ipynb
```


### 🔁 Steps Performed in the Notebook

The notebook performs the following steps:

1. Loads `medidas.csv` with distance data \( d_1, d_2, d_3(t) \)
2. Solves the trilateration equations using the **Newton–Raphson method**
3. Saves the reconstructed positions to `posiciones relativas.csv`
4. Plots convergence per iteration
5. Applies **cubic splines** to smooth the trajectory and compares it to the real one
6. Saves:
   - `trayectoria_interpolada.csv`
   - Figures:
     - `posiciones.png`
     - `trayectoria_real_interpolada.png`
     - `evolucion_error.png`

---

### 📊 4. Results and Figures

All plots and CSV files generated are saved in the same directory as each script, making them easy to reference in the report.  
Figures mentioned in the report should be manually copied to the `figuras/` folder (not included in the repository) to keep the project structure clean and organized.


## 5. Contact

Francisco Ortega – Belén Götz*  
Métodos Numéricos y Optimización – UDESA 2024 
