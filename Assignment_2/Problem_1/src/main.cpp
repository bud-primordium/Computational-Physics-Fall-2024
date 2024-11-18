/**
 * @mainpage Root Finder
 *
 * Using various numerical methods
 *
 * @details
 * The program allows users to choose from the following root-finding algorithms:
 * - Bisection Method
 * - Hybrid Method
 * - Brent Method
 * - Ridder Method
 * - Newton-Raphson Method
 *
 * Users can either use default parameters or customize them according to their needs. The program outputs include the root of the function, the number of iterations, and detailed steps.
 *
 * Additionally, the program offers options to compare all the methods and display their performance side by side.
 *
 * ## Key Features
 * - Implements five distinct root-finding algorithms
 * - Interactive user interface for method selection
 * - Customizable parameters such as tolerance and initial guesses
 * - Displays detailed performance metrics and results
 * - Provides comparative analysis across the algorithms
 *
 * ## How to Use
 * 1. Run the program.
 * 2. Select a root-finding algorithm or choose to compare all methods.
 * 3. Enter custom parameters or use the default values.
 * 4. View the results and performance metrics.
 */

/**
 * @file main.cpp
 * @author
 * Gilbert Young
 * @date
 * 2024/09/19
 * @brief
 * The main entry point for the Root-Finding Algorithms Solver project.
 */

#include "functions.h"
#include "methods.h"
#include "plotting.h"
#include "utils.h"
#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <iomanip>

int main()
{
    // Plot the function once at the beginning with range [-3, 3]
    plot_function(-3.0L, 3.0L, -10.0L, 10.0L);

    char choice;
    do
    {
        long double a = 0.0L, b = 0.0L, x0 = 0.0L, tol = 1e-14L;
        std::string method_name;

        // Get user input for method selection
        get_user_input(a, b, x0, method_name, tol);

        int max_iter = 1000;

        // Map of methods excluding Newton-Raphson and Compare All Methods
        std::map<std::string, std::function<long double(long double, long double, long double, int, std::vector<std::string> &, int)>> methods = {
            {"Bisection Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
             {
                 return bisection(a, b, tol, max_iter, iterations, decimal_places);
             }},
            {"Hybrid Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
             {
                 return hybrid_method(a, b, tol, max_iter, iterations, decimal_places);
             }},
            {"Brent Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
             {
                 return brent_method(a, b, tol, max_iter, iterations, decimal_places);
             }},
            {"Ridder Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
             {
                 return ridder_method(a, b, tol, max_iter, iterations, decimal_places);
             }}};

        if (method_name == "Newton-Raphson Method")
        {
            std::vector<std::string> iterations;
            long double root = newton_raphson(x0, tol, max_iter, iterations, calculate_decimal_places(tol));
            RootInfo info{root, static_cast<int>(iterations.size()), calculate_decimal_places(tol)};
            summary[method_name].emplace_back(info);

            // Display results
            std::cout << "\nMethod: " << method_name << "\n";
            std::cout << "Initial guess: x0 = " << std::fixed << std::setprecision(info.decimal_places) << x0 << "\n";
            std::cout << "Root: " << std::fixed << std::setprecision(info.decimal_places) << root << "\n";
            std::cout << "Iterations:\n";
            for (const auto &iter : iterations)
            {
                std::cout << iter << "\n";
            }
            std::cout << "Iterations Count: " << iterations.size() << "\n";
        }
        else if (method_name == "Problem Steps Mode")
        {
            // Run the problem steps
            run_problem_steps();
        }
        else if (method_name == "Compare All Methods")
        {
            // Run the comparison
            compare_all_methods();
        }
        else
        {
            // Get the method function
            auto it = methods.find(method_name);
            if (it != methods.end() && it->second != nullptr)
            {
                run_method_user_selection(method_name, it->second, a, b, tol, max_iter);
            }
            else
            {
                std::cerr << "Method not found or not implemented.\n";
            }
        }

        // Output summary of all results
        if (method_name != "Problem Steps Mode" && method_name != "Compare All Methods")
        {
            std::cout << "\n--- Summary of All Results ---\n";
            for (const auto &method : summary)
            {
                std::cout << "\nMethod: " << method.first << "\n";
                int idx = 1;
                for (const auto &info : method.second)
                {
                    std::cout << "  Root " << idx++ << ": " << std::fixed << std::setprecision(info.decimal_places) << info.root
                              << " | Iterations: " << info.iterations << "\n";
                }
            }
            // Clear summary for next run
            summary.clear();
        }

        // Ask user if they want to run again
        std::cout << "\nDo you want to run the program again? (y/n): ";
        std::cin >> choice;

    } while (choice == 'y' || choice == 'Y');

    // Pause and wait for user input before exiting
    std::cout << "\nPress Enter to exit...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    return 0;
}
