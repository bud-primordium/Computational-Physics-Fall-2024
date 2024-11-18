/*
@Author: Gilbert Young
@Time: 2024/09/19 08:56
@File_name: methods.h
@Description:
Header file containing declarations of the optimization methods:
1. steepestDescent
2. conjugateGradient
3. simulatedAnnealing
4. geneticAlgorithm
*/

#ifndef METHODS_H
#define METHODS_H

#include "structs.h"

// Function prototypes for optimization methods
Result steepestDescent(double x0, double y0, double alpha, int maxIter, double tol);
Result conjugateGradient(double x0, double y0, int maxIter, double tol);
Result simulatedAnnealing(double x0, double y0, double T0, double Tmin, double alpha, int maxIter);
Result geneticAlgorithm(int populationSize, int generations, double mutationRate, double crossoverRate);

#endif // METHODS_H
