/*
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: plotting.h
@IDE: VSCode
@Formatter: Clang-Format
@Description: Declaration of function to plot f(x) on a grid.
*/

#ifndef PLOTTING_H
#define PLOTTING_H

void plot_function(long double x_min, long double x_max, long double y_min, long double y_max,
                   int width = 60, int height = 20, long double label_interval = 1.0);

#endif // PLOTTING_H
