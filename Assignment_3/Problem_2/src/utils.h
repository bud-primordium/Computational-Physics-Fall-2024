/**
 * @author Gilbert Young
 * @date 2024/09/25
 * @file utils.h
 * @brief Utility functions for matrix initialization and display.
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <chrono>

/**
 * @brief Initializes the matrix by reading from a `.in` file.
 *
 * @param m Reference to the matrix to be initialized.
 * @param filename Name of the input file.
 * @param rows Reference to store the number of rows.
 * @param cols Reference to store the number of columns.
 * @return true If the matrix was successfully initialized.
 * @return false If there was an error during initialization.
 */
bool InitMatrix(std::vector<std::vector<double>> &m, const std::string &filename, int &rows, int &cols);

/**
 * @brief Displays the system of linear equations.
 *
 * @param m The matrix representing the system.
 * @param rows Number of equations.
 * @param cols Number of variables plus one (for constants).
 */
void ShowEquations(const std::vector<std::vector<double>> &m, int rows, int cols);

/**
 * @brief Checks the consistency of the system of equations.
 *
 * @param m The matrix representing the system.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return true If the system is consistent.
 * @return false If the system is inconsistent.
 */
bool CheckConsistency(const std::vector<std::vector<double>> &m, int rows, int cols);

/**
 * @brief Displays the unique solution.
 *
 * @param solution The solution vector.
 */
void DisplaySolution(const std::vector<double> &solution);

// Timing functions
std::chrono::steady_clock::time_point StartTimer();
void StopTimer(const std::chrono::steady_clock::time_point &start);

#endif // UTILS_H
