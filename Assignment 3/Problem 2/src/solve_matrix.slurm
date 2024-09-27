#!/bin/bash
#SBATCH -J Matrix_Solver           # 作业名称
#SBATCH -N 1                       # 使用1个节点
#SBATCH --ntasks-per-node=1        # 每个节点只运行1个任务
#SBATCH -p chu                     # 指定分区chu
#SBATCH -t 1-00:00:00              # 运行时间上限为1天
#SBATCH --exclusive                # 独占节点运行

# 设置线程数和其他资源
ulimit -s unlimited
export OMP_NUM_THREADS=1

# 加载Anaconda环境
source /home/wangsichao/anaconda3/bin/activate
conda activate base             # 激活base

# 运行Python脚本进行矩阵求解，并将日志输出保存
python solve_matrix.py | tee log000
