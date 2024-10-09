# Assignment 3 - Problem 3: Schrödinger Equation Solver

这是一个 Julia 项目，用于求解一维 Schrödinger 方程。该项目实现了基于高斯基函数的解法，并通过用户选择不同的势能函数来观察不同的解。

## 项目结构

- **`main.jl`**：主入口文件，包含用户接口以选择不同的势能函数。
- **`methods.jl`**：包含求解 Schrödinger 方程的方法。
- **`utils.jl`**：包含波函数的归一化工具。
- **`interaction.jl`**：处理势能函数的定义和相关计算。

## 交互式体验

点击以下按钮可在 Binder 上运行该项目，体验交互式求解 Schrödinger 方程的过程：

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bud-primordium/Computational-Physics-Fall-2024/binder_3_3?labpath=binder_3_3%2Fstart_pluto.jl)

## 使用方法

1. 选择不同的势能函数并运行代码。
2. 查看输出的能级，并选择是否可视化波函数。
