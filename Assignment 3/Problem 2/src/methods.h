/**
 * @author Gilbert Young
 * @date 2024/09/25
 * @file methods.h
 * @brief Core computational functions for solving linear systems.
 */

#ifndef METHODS_H
#define METHODS_H

#include <vector>

/**
 * @brief Performs Gaussian elimination on the matrix.
 *
 * @param m Reference to the matrix to be modified.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return int Number of row exchanges performed.
 */
int GaussianElimination(std::vector<std::vector<double>> &m, int rows, int cols);

/**
 * @brief Determines the rank of the matrix.
 *
 * @param m The matrix.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return int The rank of the matrix.
 */
int DetermineRank(const std::vector<std::vector<double>> &m, int rows, int cols);

/**
 * @brief Performs back-substitution to find the unique solution.
 *
 * @param m The upper triangular matrix after Gaussian elimination.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param solution Reference to store the solution vector.
 * @return true If a unique solution exists.
 * @return false If the system is inconsistent.
 */
bool BackSubstitution(const std::vector<std::vector<double>> &m, int rows, int cols, std::vector<double> &solution);

/**
 * @brief Displays the general solution for systems with infinitely many solutions.
 *
 * @param m The matrix after Gaussian elimination.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param rank The rank of the matrix.
 */
void ShowGeneralSolution(const std::vector<std::vector<double>> &m, int rows, int cols, int rank);

#endif // METHODS_H
