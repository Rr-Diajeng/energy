# Soft Computing Assignment 3  
# Real-World Optimization via Nature-Inspired Computing  
# PV–BESS Sizing Optimization using GA and PSO

# 1. Project Description
This project solves a real-world optimization problem for sizing an off-grid Photovoltaic (PV) and Battery Energy Storage System (BESS). The objective is to minimize the total lifecycle cost while satisfying energy demand and battery State of Charge (SoC) constraints.

Two nature-inspired optimization algorithms are implemented and compared:

1. Genetic Algorithm (GA)
2. Particle Swarm Optimization (PSO)

The optimization determines two main decision variables:
- `CAPPV`: PV system capacity
- `EBESS`: Battery Energy Storage System capacity

The search space for both variables is:
CAPPV ∈ [100, 20000]
EBESS ∈ [100, 20000]

The main battery State of Charge constraint is:
0.2 ≤ SoC ≤ 0.8

The report compares GA and PSO based on convergence behavior, total cost, PV capacity, BESS capacity, and sensitivity analysis.

# 2. Files Included
The project folder should contain the following files:

├── final.ipynb
├── data_with_datetime.csv
├── README.md
└── output_convergence/
└── output_sensitivity/

Description:
- `final.ipynb`  
  The main Jupyter Notebook containing the implementation of GA, PSO, fitness function, constraint handling, sensitivity analysis, and visualization.

- `data_with_datetime.csv`  
  Input data used in the optimization. The dataset should include solar irradiance and electricity demand data.

- `README.md`  
  Instructions for running the program.

- `output_convergence`  
  Folder for saving generated results, plots, or exported files if needed.

- `output_sensitivity`  
  Folder for saving sensitivity analysis.

# 3. Required Software
Before running the program, make sure the following software is installed:
- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- pip or Anaconda

Recommended environment:
Python >= 3.9
Jupyter Notebook
NumPy
Pandas
Matplotlib

# 4. Required Python Libraries
Install the required libraries using:
pip install numpy pandas matplotlib

If using Anaconda, the libraries can also be installed using:
conda install numpy pandas matplotlib

# 5. Dataset Requirements
The input dataset should contain hourly data related to solar generation and electricity demand. The main input columns include:
datetime
SG(t)
LOAD(kW)

Where: 
- `datetime` represents the time index.
- `SG(t)` represents the solar generation at time t.
- `LOAD(kW)` represents the electricity load demand.

Make sure the dataset file name and column names match the code. If the code uses a different file name, update the dataset path in the notebook before running.

Example:
data = pd.read_csv("data_with_datetime.csv")

# 6. How to Run the Program
# Step 1: Open the Project Folder

Open the folder containing the notebook and dataset.

Example:
cd softcomp-energy

# Step 2: Start Jupyter Notebook

Run:
jupyter notebook

or:
jupyter lab

# Step 3: Open the Notebook

Open the file:
final.ipynb

# Step 4: Run All Cells

Run all cells sequentially from top to bottom.

In Jupyter Notebook:
Kernel > Restart & Run All

This will execute:
1. Import libraries
2. Load dataset
3. Define parameters
4. Run Genetic Algorithm
5. Run Particle Swarm Optimization
6. Perform sensitivity analysis
7. Generate convergence plots
8. Compare GA and PSO results

# 7. Main Parameters
# Genetic Algorithm Parameters
| Parameter                       | Value                           |
|---------------------------------|---------------------------------|
| Population size                 | 100                             |
| Number of generations           | 100                             |
| Crossover rate                  | 0.8                             |
| Mutation rate                   | 0.1                             |
| Chromosome length per variable  | 15 bits                         |
| Selection method                | Fitness-inversed roulette wheel |
| Feasibility enforcement         | Reject-and-retry                |

# Particle Swarm Optimization Parameters
| Parameter                 | Value        |
|---------------------------|--------------|
| Number of particles       | 100          |
| Number of iterations      | 100          |
| Inertia weight            | 0.7          |
| Cognitive coefficient, C1 | 1.5          |
| Social coefficient, C2    | 1.5          |
| Maximum velocity          | 1000         |
| Position bounds           | [100, 20000] |

# 8. Output Results
After running the program, the notebook will generate several outputs, including:
1. Best fitness value
2. Best PV capacity, CAPPV
3. Best BESS capacity, EBESS
4. Total cost
5. GA convergence curve
6. PSO convergence curve
7. Sensitivity analysis for mutation rate
8. Sensitivity analysis for crossover rate
9. GA and PSO comparison plots

The main comparison includes:
- Fitness convergence
- PV capacity convergence
- BESS capacity convergence
- Total cost convergence

# 9. Reproducibility Notes
GA and PSO are stochastic algorithms. This means the results may be slightly different each time the program is run because the initial population, particle positions, crossover, mutation, and velocity updates involve random processes.

To improve reproducibility, use a fixed random seed before running the algorithms:

```python
import random

random.seed(42)
```

If a fixed seed is not used, the general trend should remain similar, but the exact numerical results may differ.

# 10. Important Notes
- Make sure the dataset path is correct before running the notebook.
- Make sure all required libraries are installed.
- Run the notebook from the first cell to the last cell.
- Do not run later cells before initialization cells.
- If an error occurs, restart the kernel and run all cells again.
- Since GA and PSO are stochastic, small differences in output are normal.

# 11. Expected Result Summary
Based on the experiment, GA can produce a feasible PV–BESS sizing solution, but it tends to converge early and may become trapped in a local optimum. PSO shows better performance in this specific experiment because it achieves a lower total cost and avoids excessive PV sizing.

However, following the No Free Lunch theorem, PSO should not be considered universally better than GA. Its better performance only applies to this specific PV–BESS sizing problem, dataset, parameter setting, and experimental configuration.

# 12. Authors
Group 9 - Soft Computing Assignment 3
- D11401811 - Darsih Idayani
- M11401836 - Hendiana
- M11401843 - Davian Benito
- M11401854 - RR. Diajeng Alfisyahrinnisa Anandha
