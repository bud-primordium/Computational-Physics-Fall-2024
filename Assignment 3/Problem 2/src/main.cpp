/**
 * @mainpage Gaussian Elimination Solver
 *
 * This project solves systems of linear equations using Gaussian elimination.
 *
 * @details
 * The program reads matrices from `.in` files, performs Gaussian elimination with partial pivoting,
 * determines the rank and consistency of the system, and displays the solution. It allows multiple runs
 * and interacts with the user for input and exit control.
 *
 * ## Features
 * - Gaussian elimination with partial pivoting
 * - Rank determination and consistency check
 * - Handles cases with no solution, unique solution, or infinitely many solutions
 *
 * ## Usage
 * 1. Provide a matrix in an `.in` file.
 * 2. The program reads the matrix and applies Gaussian elimination.
 * 3. The user can run the program multiple times.
 */

/**
 * @file main.cpp
 * @author Gilbert Young
 * @date 2024/09/25
 * @brief Entry point for the Gaussian Elimination Solver project.
 */

#include <iostream>
#include <string>
#include <vector>
#include <cmath> // Included for fabs function
#include <filesystem>
#include <limits> // For std::numeric_limits
#include "utils.h"
#include "methods.h"
#include "interaction.h"

using namespace std;

int main()
{
    char choice;
    do
    {
        string selected_file = SelectInputFile();
        if (selected_file.empty())
        {
            return 1; // File selection failed
        }

        vector<vector<double>> matrix;
        int rows, cols;
        if (!InitMatrix(matrix, selected_file, rows, cols))
        {
            return 1; // Matrix initialization failed
        }

        ShowEquations(matrix, rows, cols);
        cout << "Starting Gaussian elimination process..." << endl;
        int exchange_count = GaussianElimination(matrix, rows, cols);
        cout << "Gaussian elimination completed." << endl
             << endl;

        int rank = DetermineRank(matrix, rows, cols);
        bool consistent = CheckConsistency(matrix, rows, cols);

        if (!consistent)
        {
            cout << "The system of equations is inconsistent and has no solution." << endl;
        }
        else if (rank < (cols - 1))
        {
            ShowGeneralSolution(matrix, rows, cols, rank);
        }
        else
        {
            vector<double> solution;
            bool solvable = BackSubstitution(matrix, rows, cols, solution);
            if (solvable)
            {
                DisplaySolution(solution);
            }
            else
            {
                cout << "The system of equations is inconsistent and has no solution." << endl;
            }
        }

        choice = AskRunAgain();

    } while (choice == 'y' || choice == 'Y');

    WaitForExit();
    return 0;
}
