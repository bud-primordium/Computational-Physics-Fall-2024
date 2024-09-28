---
project: 24-Game Solver
summary:  An enhanced version of the 24-game solver using recursive search and pruning. 
src_dir: ./src
output_dir: ./doc
project_github: https://github.com/bud-primordium/Computational-Physics-Fall-2024/tree/main/Assignment%201/Problem%202
project_download: https://github.com/bud-primordium/Computational-Physics-Fall-2024/releases/tag/game24
author: Gilbert Young
author_description: An atypical physics student
github: https://github.com/bud-primordium
email: gilbertyoung0015@gmail.com
version: 2.0
year: 2024
fpp_extensions: fpp
preprocess: true
predocmark: >
docmark_alt: #
predocmark_alt: <
source: true
graph: true
coloured_edges: true
search: true
exclude: src/game24_promax.f90
---

## Overview

The **24-Game Solver** is designed to solve the classic 24-point game by using recursive search with pruning techniques. It also includes OpenMP parallelization to leverage multi-core processors and provides a progress bar for real-time monitoring during the search process.

Users can either use the default settings or customize them according to their preferences. The program outputs include:

- The first valid solution to the 24-game.
- Detailed recursive steps and expressions for the solution.

Additionally, the solver supports varying numbers of inputs (up to 8) and halts the search as soon as a valid solution is found.

## Features

- **Recursive Search with Pruning**: The solver efficiently explores number combinations to reach the target value of 24.
- **Progress Bar**: Visual feedback is provided during the search process, showing the percentage of progress.
- **OpenMP Parallelization**: The solver utilizes OpenMP to speed up the recursive search on multi-core systems.
- **Commutative Operation Optimization**: Optimizes operations such as addition and multiplication to avoid redundant calculations.
- **Supports Up to 8 Inputs**: Flexible input options for solving the 24-game with different numbers of inputs.
- **Automatic Halting**: The solver stops as soon as it finds the first valid solution, saving computational resources.

## How to Use

1. **Compile the Program**  
   Ensure you have a Fortran compiler that supports OpenMP (e.g., `gfortran`). Compile the program with OpenMP support:

   ```bash
   gfortran -fopenmp -O3 -march=native -o game24_ultra src/game24_ultra.f90
   ```

   @note  
   OpenMP is essential for maximizing the performance of the solver. It's highly recommended to compile the program with optimization flags like `-O3` for maximum optimization and `-march=native` to leverage the specific features of your CPU architecture.
   @endnote

2. **Run the Program**  
   Execute the compiled program:

   ```bash
   ./game24_ultra
   ```

3. **Follow the Prompts**  
   - **Enter the Number of Inputs**: Specify how many numbers (between 1 and 8) you want to use.
   - **Provide Input Numbers or Card Values**: Input the numbers or card values (`A=1`, `J=11`, `Q=12`, `K=13`).
   - **View the Solution**: The program will display the first valid solution found or notify you if no solution exists.
