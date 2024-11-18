/**
 * @author Gilbert Young
 * @date 2024/09/25
 * @file methods.h
 * @brief Core computational functions for solving linear systems.
 *
 * This header declares functions for Gaussian elimination with partial pivoting,
 * back-substitution, rank determination, and displaying general solutions.
 *
 */

#ifndef METHODS_H
#define METHODS_H

#include <vector>
#include <string>

/**
 * @brief Performs Gaussian elimination on the augmented matrix with partial pivoting.
 *
 * @param m Reference to the augmented matrix [A|b] to be modified.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix (including augmented column).
 * @return int Number of row exchanges performed during elimination.
 */
int GaussianElimination(std::vector<std::vector<double>> &m, int rows, int cols);

/**
 * @brief Determines the rank of the coefficient matrix A (excluding augmented column).
 *
 * @param m The augmented matrix [A|b].
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix (including augmented column).
 * @return int The rank of the matrix A.
 */
int DetermineRank(const std::vector<std::vector<double>> &m, int rows, int cols);

/**
 * @brief Performs back-substitution to find the solution vector.
 *
 * @param m The upper triangular matrix after Gaussian elimination.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix (including augmented column).
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
 * @param cols Number of columns in the matrix (including augmented column).
 * @param rank The rank of the coefficient matrix A.
 */
void ShowGeneralSolution(const std::vector<std::vector<double>> &m, int rows, int cols, int rank);

/**
 * @brief Identifies the pivot columns in the matrix.
 *
 * @param m The matrix after Gaussian elimination.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix (including augmented column).
 * @return std::vector<int> A vector containing the indices of the pivot columns.
 */
std::vector<int> IdentifyPivots(const std::vector<std::vector<double>> &m, int rows, int cols);

#endif // METHODS_H
