/*
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: utils.cpp
@IDE: VSCode
@Formatter: Clang-Format
@Description: Implementation of utility functions for running methods and handling user input.
*/

#include "utils.h"
#include "methods.h"
#include "functions.h"
#include "plotting.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <limits>

std::map<std::string, std::vector<RootInfo>> summary;

// Function to run the method and display results
void run_method(const std::string &method_name,
                std::function<long double(long double, long double, long double, int, std::vector<std::string> &, int)> method_func,
                long double a, long double b, long double tol, int max_iter,
                int decimal_places)
{
    std::vector<std::string> iterations;
    long double root = method_func(a, b, tol, max_iter, iterations, decimal_places);

    RootInfo info{root, static_cast<int>(iterations.size()), decimal_places};
    summary[method_name].emplace_back(info);

    // Display results
    std::cout << "\nMethod: " << method_name << "\n";
    if (method_name == "Newton-Raphson Method")
    {
        std::cout << "Initial guess: x0 = " << std::fixed << std::setprecision(decimal_places) << a << "\n";
    }
    else
    {
        std::cout << "Interval: [" << std::fixed << std::setprecision(2) << a << ", " << b << "]\n";
    }
    std::cout << "Root: " << std::fixed << std::setprecision(decimal_places) << root << "\n";
    std::cout << "Iterations:\n";
    for (const auto &iter : iterations)
    {
        std::cout << iter << "\n";
    }
    std::cout << "Iterations Count: " << iterations.size() << "\n";
}

// Function to run the problem steps
void run_problem_steps()
{
    // Define intervals for three roots
    std::vector<std::pair<long double, long double>> intervals = {
        {-3.0L, -2.0L}, // Negative root
        {0.0L, 1.0L},   // First positive root
        {1.0L, 3.0L}    // Second positive root
    };

    // Vector to store roots found in part (i)
    std::vector<long double> found_roots;

    // Define tolerances and maximum iterations
    long double tol_bisection = 1e-4L; // 4 decimal places
    long double tol_newton = 1e-14L;   // 14 decimal places
    long double tol_hybrid = 1e-14L;   // 14 decimal places
    int max_iter = 1000;

    std::cout << "\n--- Problem Steps Execution ---\n";

    // Part (i): Bisection Method to find three roots to 4 decimal places
    std::cout << "\nPart (i): Bisection Method to find roots to 4 decimal places\n";
    for (const auto &interval : intervals)
    {
        std::vector<std::string> iterations;
        long double root = bisection(interval.first, interval.second, tol_bisection, max_iter, iterations, 4);
        RootInfo info{root, static_cast<int>(iterations.size()), 4};
        summary["Bisection Method"].emplace_back(info);

        // Store the found root
        found_roots.emplace_back(root);

        std::cout << "Root in [" << std::fixed << std::setprecision(2) << interval.first << ", " << interval.second << "]: "
                  << std::fixed << std::setprecision(4) << root << "\n";
        std::cout << "Iterations: " << iterations.size() << "\n";
    }

    // Part (ii): Newton-Raphson Method to refine the three roots to 14 decimal places
    std::cout << "\nPart (ii): Newton-Raphson Method to refine roots to 14 decimal places\n";
    for (auto &x0 : found_roots)
    {
        std::vector<std::string> iterations;
        long double root = newton_raphson(x0, tol_newton, max_iter, iterations, 14);
        RootInfo info{root, static_cast<int>(iterations.size()), 14};
        summary["Newton-Raphson Method"].emplace_back(info);

        std::cout << "Refined root starting from " << std::fixed << std::setprecision(4) << x0 << ": "
                  << std::fixed << std::setprecision(14) << root << "\n";
        std::cout << "Iterations: " << iterations.size() << "\n";
    }

    // Part (iii): Hybrid Method to find three roots to 14 decimal places
    std::cout << "\nPart (iii): Hybrid Method to find roots to 14 decimal places\n";
    for (const auto &interval : intervals)
    {
        std::vector<std::string> iterations;
        long double root = hybrid_method(interval.first, interval.second, tol_hybrid, max_iter, iterations, 14);
        RootInfo info{root, static_cast<int>(iterations.size()), 14};
        summary["Hybrid Method"].emplace_back(info);

        std::cout << "Root in [" << std::fixed << std::setprecision(2) << interval.first << ", " << interval.second << "] (Hybrid): "
                  << std::fixed << std::setprecision(14) << root << "\n";
        std::cout << "Iterations: " << iterations.size() << "\n";
    }

    // Output summary of results for problem steps
    std::cout << "\n--- Summary of Problem Steps Results ---\n";
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

// Function to compare all methods
void compare_all_methods()
{
    // Define intervals for three roots
    std::vector<std::pair<long double, long double>> intervals = {
        {-3.0L, -2.0L}, // Negative root
        {0.0L, 1.0L},   // First positive root
        {1.0L, 3.0L}    // Second positive root
    };

    // Define tolerances and maximum iterations
    long double tol = 1e-14L; // 14 decimal places
    int max_iter = 1000;

    // List of methods to compare
    std::vector<std::pair<std::string, std::function<long double(long double, long double, long double, int, std::vector<std::string> &, int)>>> methods = {
        {"Bisection Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
         {
             return bisection(a, b, tol, max_iter, iterations, decimal_places);
         }},
        {"Newton-Raphson Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
         {
             // For Newton-Raphson, use the midpoint as the initial guess
             long double initial_guess = (a + b) / 2.0L;
             return newton_raphson(initial_guess, tol, max_iter, iterations, decimal_places);
         }},
        {"Hybrid Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
         {
             return hybrid_method(a, b, tol, max_iter, iterations, decimal_places);
         }},
        {"Brent's Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
         {
             return brent_method(a, b, tol, max_iter, iterations, decimal_places);
         }},
        {"Ridder's Method", [](long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places) -> long double
         {
             return ridder_method(a, b, tol, max_iter, iterations, decimal_places);
         }}};

    // Store comparison results
    std::map<std::string, std::vector<RootInfo>> comparison_results;

    // Run each method for each interval
    for (const auto &method : methods)
    {
        for (const auto &interval : intervals)
        {
            std::vector<std::string> iterations;
            long double root = method.second(interval.first, interval.second, tol, max_iter, iterations, 15); // 1e-14 -> 15 decimal places
            RootInfo info{root, static_cast<int>(iterations.size()), 15};
            comparison_results[method.first].emplace_back(info);
        }
    }

    // Display comparison table
    std::cout << "\n--- Comparison of All Methods (Precision: 1e-14) ---\n\n";

    // Table header
    std::cout << std::left << std::setw(25) << "Method"
              << std::setw(30) << "Root 1 (-3,-2)"
              << std::setw(15) << "Iterations"
              << std::setw(30) << "Root 2 (0,1)"
              << std::setw(15) << "Iterations"
              << std::setw(30) << "Root 3 (1,3)"
              << std::setw(15) << "Iterations" << "\n";

    // Separator
    std::cout << std::string(130, '-') << "\n";

    // Table rows
    for (const auto &method : methods)
    {
        std::cout << std::left << std::setw(25) << method.first;
        for (size_t i = 0; i < intervals.size(); ++i)
        {
            if (comparison_results[method.first][i].root != comparison_results[method.first][i].root)
            {
                // Check for NAN
                std::cout << std::left << std::setw(30) << "N/A"
                          << std::left << std::setw(15) << "N/A";
            }
            else
            {
                std::cout << std::left << std::setw(30) << std::fixed << std::setprecision(15) << comparison_results[method.first][i].root
                          << std::left << std::setw(15) << comparison_results[method.first][i].iterations;
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nNote: Precision is set to 1e-14, output displays 15 decimal places.\n\n";
}

// Function to get user input
void get_user_input(long double &a, long double &b, long double &x0, std::string &method_name, long double &tol)
{
    // List of available methods
    std::vector<std::string> available_methods = {"Bisection Method", "Hybrid Method", "Brent Method", "Ridder Method", "Newton-Raphson Method", "Problem Steps Mode", "Compare All Methods"};

    // Display available methods
    std::cout << "\nAvailable methods:\n";
    for (size_t i = 0; i < available_methods.size(); ++i)
    {
        std::cout << i + 1 << ". " << available_methods[i] << "\n";
    }

    // Prompt user to select a method
    int method_choice;
    std::cout << "Select a method (1-" << available_methods.size() << "): ";
    std::cin >> method_choice;
    while (method_choice < 1 || method_choice > static_cast<int>(available_methods.size()))
    {
        std::cout << "Invalid choice. Please select a method (1-" << available_methods.size() << "): ";
        std::cin >> method_choice;
    }
    method_name = available_methods[method_choice - 1];

    if (method_name == "Newton-Raphson Method")
    {
        // Prompt user to input initial guess x0
        std::cout << "Enter initial guess x0: ";
        std::cin >> x0;
    }
    else if (method_name != "Problem Steps Mode" && method_name != "Compare All Methods")
    {
        // Prompt user to input interval [a, b]
        std::cout << "Enter interval [a, b]:\n";
        std::cout << "a = ";
        std::cin >> a;
        std::cout << "b = ";
        std::cin >> b;
        while (a >= b)
        {
            std::cout << "Invalid interval. 'a' should be less than 'b'. Please re-enter:\n";
            std::cout << "a = ";
            std::cin >> a;
            std::cout << "b = ";
            std::cin >> b;
        }
    }

    if (method_name != "Problem Steps Mode" && method_name != "Compare All Methods")
    {
        // Prompt user to input desired precision
        std::cout << "Enter desired precision (e.g., 1e-14, up to 1e-16): ";
        std::cin >> tol;
        const long double min_tol = 1e-16L;
        const long double max_tol = 1e-4L;
        while (tol < min_tol || tol > max_tol)
        {
            std::cout << "Precision out of bounds (" << min_tol << " to " << max_tol << "). Please re-enter: ";
            std::cin >> tol;
        }
    }
}

// Function to calculate decimal places based on tolerance
int calculate_decimal_places(long double tol)
{
    if (tol <= 0)
        return 0;
    return static_cast<int>(ceil(-log10(tol))) + 1;
}

// Function to run the method and display results (for user-selected methods)
void run_method_user_selection(const std::string &method_name,
                               std::function<long double(long double, long double, long double, int, std::vector<std::string> &, int)> method_func,
                               long double a, long double b, long double tol, int max_iter)
{
    int decimal_places = calculate_decimal_places(tol);
    run_method(method_name, method_func, a, b, tol, max_iter, decimal_places);
}
