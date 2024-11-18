/*
@Author: Gilbert Young
@Time: 2024/09/19 08:56
@File_name: functions.h
@Description:
Header file containing function declarations for the mathematical functions used in the optimization algorithms:
1. functionToMinimize: the function to be minimized.
2. computeGradient: computes the gradient of the function.
3. lineSearchBacktracking: performs a backtracking line search for the Conjugate Gradient method.
*/

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

double functionToMinimize(double x, double y);
void computeGradient(double x, double y, double &dx, double &dy);
double lineSearchBacktracking(double x, double y, double dx, double dy, double alpha_init = 1.0, double rho = 0.5, double c = 1e-4);

#endif // FUNCTIONS_H
