/*
@Author: Gilbert Young
@Time: 2024/09/19 08:56
@File_name: structs.h
@Description:
Header file containing the definitions of data structures used in the optimization algorithms:
1. Individual: structure to store coordinates and fitness for the Genetic Algorithm.
2. DefaultParameters: structure for default parameters for all algorithms.
3. Result: structure to store the optimization results.
*/

#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>

// Structure to store coordinates and fitness for Genetic Algorithm
struct Individual
{
    double x;
    double y;
    double fitness;

    // Constructor for easier initialization
    Individual(double x_val = 0, double y_val = 0) : x(x_val), y(y_val), fitness(0) {}
};

// Structure for default parameters
struct DefaultParameters
{
    // Initial points
    double x0 = 0.0;
    double y0 = 0.0;

    // Steepest Descent parameters
    double alpha_sd = 0.0050;
    double tol_sd = 1e-8;
    int maxIter_sd = 100000;

    // Conjugate Gradient parameters
    double tol_cg = 1e-8;
    int maxIter_cg = 100000;

    // Simulated Annealing parameters
    double T0_sa = 2000.0;
    double Tmin_sa = 1e-8;
    double alpha_sa = 0.99;
    int maxIter_sa = 200000;

    // Genetic Algorithm parameters
    int populationSize_ga = 100;
    int generations_ga = 5000;
    double mutationRate_ga = 0.02;
    double crossoverRate_ga = 0.8;
};

// Structure to store optimization results
struct Result
{
    double x;
    double y;
    double f;
    int iterations;
    double duration; // in seconds
};

#endif // STRUCTS_H
