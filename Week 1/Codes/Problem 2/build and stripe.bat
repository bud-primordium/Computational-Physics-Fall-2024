@echo off
REM 编译 game24_promax.f90，使用优化级别 O，生成可执行文件 promax_test
echo Compiling game24_promax.f90 with optimization O...
gfortran -O game24_promax.f90 -o promax_test

REM 编译 game24_ultra.f90，使用优化级别 O，并启用 OpenMP，生成 ultra_test
echo Compiling game24_ultra.f90 with optimization O and OpenMP support...
gfortran -O -fopenmp game24_ultra.f90 -o ultra_test

REM 使用优化级别 O3，32位模式编译 game24_promax.f90，生成可执行文件 game24_promax
echo Compiling game24_promax.f90 with optimization O3, 32-bit mode...
gfortran -O3 -m32 game24_promax.f90 -o game24_promax

REM 使用优化级别 O3，x86-64架构编译 game24_promax.f90，生成可执行文件 game24_promax_x64
echo Compiling game24_promax.f90 with optimization O3, x86-64 architecture...
gfortran -O3 -march=x86-64 game24_promax.f90 -o game24_promax_x64

REM 使用优化级别 O3，32位模式并启用 OpenMP，编译 game24_ultra.f90，生成可执行文件 game24_ultra
echo Compiling game24_ultra.f90 with optimization O3, 32-bit mode and OpenMP support...
gfortran -O3 -m32 -fopenmp game24_ultra.f90 -o game24_ultra

REM 使用优化级别 O3，x86-64架构并启用 OpenMP，编译 game24_ultra.f90，生成可执行文件 game24_ultra_x64
echo Compiling game24_ultra.f90 with optimization O3, x86-64 architecture and OpenMP support...
gfortran -O3 -march=x86-64 -fopenmp game24_ultra.f90 -o game24_ultra_x64

REM 使用优化级别 O3，本地架构并启用 OpenMP，编译 game24_ultra.f90，生成本地可执行文件 game24_ultra_local
echo Compiling game24_ultra.f90 with optimization O3, native architecture and OpenMP support...
gfortran -O3 -march=native -fopenmp .\game24_ultra.f90 -o .\game24_ultra_local

REM 对当前目录下的所有 .exe 文件执行 strip 操作，去除符号表和调试信息以减小文件体积
echo Stripping all .exe files in the current directory...
for %%f in (*.exe) do strip %%f

echo Compilation and stripping process complete!
