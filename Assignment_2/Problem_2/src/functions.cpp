/*
@Author: Gilbert Young
@Time: 2024/09/19 08:56
@File_name: functions.cpp
@Description:
Implementation file containing the definitions of the mathematical functions used in the optimization algorithms.
*/

#include "functions.h"
#include <cmath>

// Function to minimize
double functionToMinimize(double x, double y)
{
    return sin(x + y) + cos(x + 2 * y);
}

// Compute gradient of the function
void computeGradient(double x, double y, double &dx, double &dy)
{
    dx = cos(x + y) - sin(x + 2 * y);
    dy = cos(x + y) - 2 * sin(x + 2 * y);
}

// Backtracking Line Search
double lineSearchBacktracking(double x, double y, double d_x, double d_y, double alpha_init, double rho, double c)
{
    double alpha = alpha_init;
    double f0 = functionToMinimize(x, y);
    double grad_dot_dir = d_x * d_x + d_y * d_y; // Since d is -grad
    while (functionToMinimize(x + alpha * d_x, y + alpha * d_y) > f0 + c * alpha * grad_dot_dir)
    {
        alpha *= rho;
        if (alpha < 1e-8)
            break;
    }
    return alpha;
}
