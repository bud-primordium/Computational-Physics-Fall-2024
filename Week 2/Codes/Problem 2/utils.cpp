/*
@Author: Gilbert Young
@Time: 2024/09/19 08:56
@File_name: utils.cpp
@Description:
Implementation file containing utility functions:
1. displayDefaultParameters: displays the default parameters for the selected algorithm.
2. compareMethods: compares all optimization methods using default parameters.
*/

#include "utils.h"
#include "methods.h"
#include <iostream>
#include <iomanip>

void displayDefaultParameters(const DefaultParameters &params, int option)
{
    if (option == 1)
    {
        std::cout << "\nCurrent Default Parameters for Steepest Descent:\n";
        std::cout << "Initial x: " << std::fixed << std::setprecision(4) << params.x0
                  << ", Initial y: " << params.y0 << "\n";
        std::cout << "Learning rate alpha: " << params.alpha_sd
                  << "\nMaximum iterations: " << params.maxIter_sd
                  << "\nTolerance: " << std::scientific << std::setprecision(3) << params.tol_sd << std::fixed << "\n";
    }
    else if (option == 2)
    {
        std::cout << "\nCurrent Default Parameters for Conjugate Gradient:\n";
        std::cout << "Initial x: " << std::fixed << std::setprecision(4) << params.x0
                  << ", Initial y: " << params.y0 << "\n";
        std::cout << "Maximum iterations: " << params.maxIter_cg
                  << "\nTolerance: " << std::scientific << std::setprecision(3) << params.tol_cg << std::fixed << "\n";
    }
    else if (option == 3)
    {
        std::cout << "\nCurrent Default Parameters for Simulated Annealing:\n";
        std::cout << "Initial x: " << std::fixed << std::setprecision(4) << params.x0
                  << ", Initial y: " << params.y0 << "\n";
        std::cout << "Initial temperature T0: " << params.T0_sa
                  << "\nMinimum temperature Tmin: " << std::scientific << std::setprecision(3) << params.Tmin_sa << std::fixed
                  << "\nCooling rate alpha: " << params.alpha_sa
                  << "\nMaximum iterations: " << params.maxIter_sa << "\n";
    }
    else if (option == 4)
    {
        std::cout << "\nCurrent Default Parameters for Genetic Algorithm:\n";
        std::cout << "Population size: " << params.populationSize_ga
                  << "\nGenerations: " << params.generations_ga
                  << "\nMutation rate: " << std::fixed << std::setprecision(4) << params.mutationRate_ga
                  << "\nCrossover rate: " << params.crossoverRate_ga << "\n";
    }
}

// Compare all methods using default parameters
void compareMethods(const DefaultParameters &params)
{
    std::cout << "\nComparing All Methods with Default Parameters...\n";

    // Display default parameters
    std::cout << "\nDefault Parameters:\n";
    std::cout << "Initial x: " << std::fixed << std::setprecision(4) << params.x0
              << ", Initial y: " << params.y0 << "\n";
    std::cout << "Steepest Descent alpha: " << params.alpha_sd
              << ", maxIter: " << params.maxIter_sd
              << ", tol: " << std::scientific << std::setprecision(3) << params.tol_sd << std::fixed << "\n";
    std::cout << "Conjugate Gradient maxIter: " << params.maxIter_cg
              << ", tol: " << std::scientific << std::setprecision(3) << params.tol_cg << std::fixed << "\n";
    std::cout << "Simulated Annealing T0: " << params.T0_sa
              << ", Tmin: " << std::scientific << std::setprecision(3) << params.Tmin_sa
              << std::fixed << ", alpha: " << params.alpha_sa
              << ", maxIter: " << params.maxIter_sa << "\n";
    std::cout << "Genetic Algorithm populationSize: " << params.populationSize_ga
              << ", generations: " << params.generations_ga
              << ", mutationRate: " << std::fixed << std::setprecision(4) << params.mutationRate_ga
              << ", crossoverRate: " << params.crossoverRate_ga << "\n";

    // Run all methods
    Result res_sd = steepestDescent(params.x0, params.y0, params.alpha_sd, params.maxIter_sd, params.tol_sd);
    Result res_cg = conjugateGradient(params.x0, params.y0, params.maxIter_cg, params.tol_cg);
    Result res_sa = simulatedAnnealing(params.x0, params.y0, params.T0_sa, params.Tmin_sa, params.alpha_sa, params.maxIter_sa);
    Result res_ga = geneticAlgorithm(params.populationSize_ga, params.generations_ga, params.mutationRate_ga, params.crossoverRate_ga);

    // Display results
    std::cout << "\nResults:\n";

    // Steepest Descent
    std::cout << "Steepest Descent Method:\n";
    std::cout << "Minimum at (" << std::fixed << std::setprecision(5) << res_sd.x << ", " << res_sd.y << ")\n";
    std::cout << "Minimum value: " << std::fixed << std::setprecision(5) << res_sd.f << "\n";
    std::cout << "Total iterations: " << res_sd.iterations << "\n";
    std::cout << "Execution Time: " << std::scientific << std::setprecision(3) << res_sd.duration << " seconds\n\n";

    // Conjugate Gradient
    std::cout << "Conjugate Gradient Method:\n";
    std::cout << "Minimum at (" << std::fixed << std::setprecision(5) << res_cg.x << ", " << res_cg.y << ")\n";
    std::cout << "Minimum value: " << std::fixed << std::setprecision(5) << res_cg.f << "\n";
    std::cout << "Total iterations: " << res_cg.iterations << "\n";
    std::cout << "Execution Time: " << std::scientific << std::setprecision(3) << res_cg.duration << " seconds\n\n";

    // Simulated Annealing
    std::cout << "Simulated Annealing:\n";
    std::cout << "Minimum at (" << std::fixed << std::setprecision(5) << res_sa.x << ", " << res_sa.y << ")\n";
    std::cout << "Minimum value: " << std::fixed << std::setprecision(5) << res_sa.f << "\n";
    std::cout << "Total iterations: " << res_sa.iterations << "\n";
    std::cout << "Execution Time: " << std::scientific << std::setprecision(3) << res_sa.duration << " seconds\n\n";

    // Genetic Algorithm
    std::cout << "Genetic Algorithm:\n";
    std::cout << "Minimum at (" << std::fixed << std::setprecision(5) << res_ga.x << ", " << res_ga.y << ")\n";
    std::cout << "Minimum value: " << std::fixed << std::setprecision(5) << res_ga.f << "\n";
    std::cout << "Total iterations: " << res_ga.iterations << "\n";
    std::cout << "Execution Time: " << std::scientific << std::setprecision(3) << res_ga.duration << " seconds\n";
}
