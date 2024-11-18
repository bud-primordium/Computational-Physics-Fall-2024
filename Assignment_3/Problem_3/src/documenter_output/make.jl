# documenter_output/make.jl

# 加载 Documenter.jl
using Documenter
include("../main.jl")  # 包含主模块但不运行主程序

# 生成文档
makedocs(
    sitename="SchrödingerSolver Documentation",
    modules=[SchrödingerSolver],
    format=Documenter.HTML(inventory_version="1.0", prettyurls=false),
    source="documenter_src",
    pages=[
        "Home" => "index.md",
        "Methods" => "methods.md",
        "Utils" => "utils.md",
        "Interaction" => "interaction.md"
    ]
)
