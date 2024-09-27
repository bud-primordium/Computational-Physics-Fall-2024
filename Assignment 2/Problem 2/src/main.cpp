/**
 * @mainpage Optimization Algorithms Solver
 *
 * Find the minimum of one multi-dims function
 *
 * @details
 * The program provides a menu for the user to select one of the following algorithms:
 * - Steepest Descent Method
 * - Conjugate Gradient Method
 * - Simulated Annealing
 * - Genetic Algorithm
 *
 * Users can choose to use default parameters or customize them. The program displays the results including the minimum point, function value, total iterations, and execution time.
 *
 * Option 5 compares all methods using default parameters, displaying their performance side by side.
 *
 * ## Features
 * - Implementation of four optimization algorithms
 * - Interactive menu for algorithm selection
 * - Customizable parameters
 * - Detailed results with performance metrics
 * - Comparative analysis of algorithms
 *
 * ## Usage
 * 1. Run the program.
 * 2. Select an optimization algorithm or choose to compare all methods.
 * 3. Opt to use default parameters or enter custom parameters.
 * 4. View the results and performance metrics.
 */

/**
 * @file main.cpp
 * @author
 * Gilbert Young
 * @date
 * 2024/09/19
 * @brief
 * Entry point for the Optimization Algorithms Solver project.
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <iomanip>
#include "methods.h"
#include "utils.h"

int main()
{
    srand(static_cast<unsigned int>(time(0)));
    char choice;
    DefaultParameters params;

    do
    {
        std::cout << "\nOptimization Algorithms Menu:\n";
        std::cout << "1. Steepest Descent Method\n";
        std::cout << "2. Conjugate Gradient Method\n";
        std::cout << "3. Simulated Annealing\n";
        std::cout << "4. Genetic Algorithm\n";
        std::cout << "5. Compare All Methods\n";
        std::cout << "Enter your choice (1-5): ";
        int option;
        std::cin >> option;

        if (option >= 1 && option <= 4)
        {
            displayDefaultParameters(params, option);
            std::cout << "\n1. Use default parameters\n";
            std::cout << "2. Customize parameters\n";
            std::cout << "Enter your choice (1-2): ";
            int subOption;
            std::cin >> subOption;

            Result res;
            if (subOption == 1)
            {
                // Run method with default parameters
                if (option == 1)
                    res = steepestDescent(params.x0, params.y0, params.alpha_sd, params.maxIter_sd, params.tol_sd);
                else if (option == 2)
                    res = conjugateGradient(params.x0, params.y0, params.maxIter_cg, params.tol_cg);
                else if (option == 3)
                    res = simulatedAnnealing(params.x0, params.y0, params.T0_sa, params.Tmin_sa, params.alpha_sa, params.maxIter_sa);
                else if (option == 4)
                    res = geneticAlgorithm(params.populationSize_ga, params.generations_ga, params.mutationRate_ga, params.crossoverRate_ga);

                // Display results
                std::cout << "\nResults:\n";
                std::cout << "Minimum at (" << std::fixed << std::setprecision(4) << res.x << ", " << res.y << ")\n";
                std::cout << "Minimum value: " << std::fixed << std::setprecision(4) << res.f << "\n";
                std::cout << "Total iterations: " << res.iterations << "\n";
                std::cout << "Execution Time: " << std::scientific << std::setprecision(3) << res.duration << " seconds\n";
            }
            else if (subOption == 2)
            {
                // Customize parameters
                std::cout << "Enter initial x (default " << std::fixed << std::setprecision(4) << params.x0 << "): ";
                std::cin >> params.x0;
                std::cout << "Enter initial y (default " << std::fixed << std::setprecision(4) << params.y0 << "): ";
                std::cin >> params.y0;

                if (option == 1)
                {
                    std::cout << "Enter learning rate alpha (default " << std::fixed << std::setprecision(4) << params.alpha_sd << "): ";
                    std::cin >> params.alpha_sd;
                    std::cout << "Enter maximum iterations (default " << params.maxIter_sd << "): ";
                    std::cin >> params.maxIter_sd;
                    std::cout << "Enter tolerance (default " << std::scientific << std::setprecision(3) << params.tol_sd << "): ";
                    std::cin >> params.tol_sd;
                    res = steepestDescent(params.x0, params.y0, params.alpha_sd, params.maxIter_sd, params.tol_sd);
                }
                else if (option == 2)
                {
                    std::cout << "Enter maximum iterations (default " << params.maxIter_cg << "): ";
                    std::cin >> params.maxIter_cg;
                    std::cout << "Enter tolerance (default " << std::scientific << std::setprecision(3) << params.tol_cg << "): ";
                    std::cin >> params.tol_cg;
                    res = conjugateGradient(params.x0, params.y0, params.maxIter_cg, params.tol_cg);
                }
                else if (option == 3)
                {
                    std::cout << "Enter initial temperature T0 (default " << std::fixed << std::setprecision(4) << params.T0_sa << "): ";
                    std::cin >> params.T0_sa;
                    std::cout << "Enter minimum temperature Tmin (default " << std::scientific << std::setprecision(6) << params.Tmin_sa << "): ";
                    std::cin >> params.Tmin_sa;
                    std::cout << "Enter cooling rate alpha (default " << std::fixed << std::setprecision(4) << params.alpha_sa << "): ";
                    std::cin >> params.alpha_sa;
                    std::cout << "Enter maximum iterations (default " << params.maxIter_sa << "): ";
                    std::cin >> params.maxIter_sa;
                    res = simulatedAnnealing(params.x0, params.y0, params.T0_sa, params.Tmin_sa, params.alpha_sa, params.maxIter_sa);
                }
                else if (option == 4)
                {
                    std::cout << "Enter population size (default " << params.populationSize_ga << "): ";
                    std::cin >> params.populationSize_ga;
                    std::cout << "Enter number of generations (default " << params.generations_ga << "): ";
                    std::cin >> params.generations_ga;
                    std::cout << "Enter mutation rate (default " << std::fixed << std::setprecision(4) << params.mutationRate_ga << "): ";
                    std::cin >> params.mutationRate_ga;
                    std::cout << "Enter crossover rate (default " << std::fixed << std::setprecision(4) << params.crossoverRate_ga << "): ";
                    std::cin >> params.crossoverRate_ga;
                    res = geneticAlgorithm(params.populationSize_ga, params.generations_ga, params.mutationRate_ga, params.crossoverRate_ga);
                }

                // Display results
                std::cout << "\nResults:\n";
                std::cout << "Minimum at (" << std::fixed << std::setprecision(4) << res.x << ", " << res.y << ")\n";
                std::cout << "Minimum value: " << std::fixed << std::setprecision(4) << res.f << "\n";
                std::cout << "Total iterations: " << res.iterations << "\n";
                std::cout << "Execution Time: " << std::scientific << std::setprecision(3) << res.duration << " seconds\n";
            }
            else
            {
                std::cout << "Invalid sub-option. Please select 1 or 2.\n";
            }
        }
        else if (option == 5)
        {
            // Compare all methods with default parameters
            compareMethods(params);
        }
        else
        {
            std::cout << "Invalid option. Please select 1-5.\n";
        }

        // Ask user if they want to run again
        std::cout << "\nDo you want to run the program again? (y/n): ";
        std::cin >> choice;
    } while (choice == 'y' || choice == 'Y');

    // Wait for user input before exiting
    std::cout << "\nPress Enter to exit...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear input buffer
    std::cin.get();                                                     // Wait for Enter key
    return 0;
}
