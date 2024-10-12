using Pkg
Pkg.precompile()
using PackageCompiler

# 生成通用 64 位 CPU 架构的应用
create_app("./project", "SchrodingerSolver";
    cpu_target="x86-64", force=true)

# 构建可执行文件
build_executable("SchrodingerSolver")
