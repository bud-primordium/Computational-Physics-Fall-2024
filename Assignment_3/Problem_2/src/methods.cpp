/**
 * @author Gilbert Young
 * @date 2024/09/25
 * @file methods.cpp
 * @brief Implementation of computational functions for solving linear systems.
 *
 * This file implements key algorithms such as Gaussian elimination with partial pivoting,
 * back-substitution, and rank determination. It also includes functionality to display
 * the general solution when the system has infinitely many solutions.
 */

#include "methods.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

using namespace std;

/// Performs partial pivoting and returns the row index with the maximum pivot.
int Pivoting(const vector<vector<double>> &m, int current_row, int total_rows)
{
    int imax = current_row;
    double max_val = fabs(m[current_row][current_row]);
    for (int i = current_row + 1; i < total_rows; i++)
    {
        double val = fabs(m[i][current_row]);
        if (val > max_val)
        {
            imax = i;
            max_val = val;
        }
    }
    return imax;
}

/// Swaps two rows in the matrix and outputs the action.
void Exchange(vector<vector<double>> &m, int row1, int row2)
{
    swap(m[row1], m[row2]);
    cout << "Swapping row " << row1 + 1 << " with row " << row2 + 1 << "." << endl;
}

/// Performs elimination on the matrix to form an upper triangular matrix.
bool Eliminate(vector<vector<double>> &m, int current_row, int total_rows, int total_cols)
{
    double pivot = m[current_row][current_row];
    if (fabs(pivot) < 1e-12)
    {
        // Pivot is too small, cannot eliminate
        return false;
    }

    for (int i = current_row + 1; i < total_rows; i++)
    {
        double factor = m[i][current_row] / pivot;
        cout << "Eliminating element in row " << i + 1 << ", column " << current_row + 1 << ":" << endl;
        cout << "Multiplying row " << current_row + 1 << " by " << fixed << setprecision(4) << factor
             << " and subtracting from row " << i + 1 << "." << endl;
        m[i][current_row] = 0.0;
        for (int j = current_row + 1; j < total_cols; j++)
        {
            m[i][j] -= factor * m[current_row][j];
        }
        cout << endl;
    }
    return true;
}

/**
 * @copydoc GaussianElimination(std::vector<std::vector<double>> &, int, int)
 */
int GaussianElimination(vector<vector<double>> &m, int rows, int cols)
{
    int exchange_count = 0;
    int n = min(rows, cols - 1); // Number of variables

    for (int k = 0; k < n; k++)
    {
        cout << "Processing column " << k + 1 << "..." << endl;

        // Find the row with the maximum pivot element
        int imax = Pivoting(m, k, rows);

        // Swap the current row with the pivot row if necessary
        if (imax != k)
        {
            Exchange(m, k, imax);
            exchange_count++;
        }
        else
        {
            cout << "No need to swap rows for column " << k + 1 << "." << endl;
        }

        // Check if pivot element is near zero (singular matrix)
        if (fabs(m[k][k]) < 1e-12)
        {
            cout << "Warning: Pivot element in row " << k + 1 << " is close to zero. The matrix may be singular." << endl;
            continue; // Skip elimination for this pivot
        }

        // Eliminate entries below the pivot
        if (!Eliminate(m, k, rows, cols))
        {
            cout << "Elimination failed for column " << k + 1 << "." << endl;
        }

        // Display current matrix state
        cout << "Current matrix state:" << endl;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                double coeff = round(m[r][c] * 1e12) / 1e12; // Handle floating-point precision
                if (fabs(coeff - round(coeff)) < 1e-12)
                {
                    cout << static_cast<long long>(round(coeff)) << "\t";
                }
                else
                {
                    cout << fixed << setprecision(2) << coeff << "\t";
                }
            }
            cout << endl;
        }
        cout << "-------------------------------------" << endl;
    }
    return exchange_count;
}

/**
 * @copydoc BackSubstitution(const std::vector<std::vector<double>> &, int, int, std::vector<double> &)
 */
bool BackSubstitution(const vector<vector<double>> &m, int rows, int cols, vector<double> &solution)
{
    solution.assign(cols - 1, 0.0);
    cout << "Starting back-substitution process..." << endl;
    for (int i = rows - 1; i >= 0; i--)
    {
        // Find the first non-zero coefficient in the row
        int pivot_col = -1;
        for (int j = 0; j < cols - 1; j++)
        {
            if (fabs(m[i][j]) > 1e-12)
            {
                pivot_col = j;
                break;
            }
        }

        if (pivot_col == -1)
        {
            if (fabs(m[i][cols - 1]) > 1e-12)
            {
                // Inconsistent equation
                return false;
            }
            else
            {
                // 0 = 0, skip
                continue;
            }
        }

        double rhs = m[i][cols - 1];
        cout << "Calculating x" << pivot_col + 1 << ":" << endl;
        for (int j = pivot_col + 1; j < cols - 1; j++)
        {
            cout << "    " << fixed << setprecision(4) << m[i][j] << " * x" << j + 1
                 << " = " << m[i][j] * solution[j] << endl;
            rhs -= m[i][j] * solution[j];
        }
        cout << "    RHS after subtraction = " << rhs << endl;
        solution[pivot_col] = rhs / m[i][pivot_col];
        cout << "    x" << pivot_col + 1 << " = " << rhs << " / " << m[i][pivot_col]
             << " = " << fixed << setprecision(4) << solution[pivot_col] << endl
             << endl;
    }
    return true;
}

/**
 * @copydoc DetermineRank(const std::vector<std::vector<double>> &, int, int)
 */
int DetermineRank(const vector<vector<double>> &m, int rows, int cols)
{
    int rank = 0;
    for (int i = 0; i < rows; i++)
    {
        bool non_zero = false;
        for (int j = 0; j < cols - 1; j++)
        {
            if (fabs(m[i][j]) > 1e-12)
            {
                non_zero = true;
                break;
            }
        }
        if (non_zero)
            rank++;
    }
    return rank;
}

/**
 * @copydoc ShowGeneralSolution(const std::vector<std::vector<double>> &, int, int, int)
 */
void ShowGeneralSolution(const vector<vector<double>> &m, int rows, int cols, int rank)
{
    cout << "The system has infinitely many solutions." << endl;
    cout << "Solution space dimension: " << (cols - 1 - rank) << endl;

    // Identify pivot columns
    vector<int> pivots = IdentifyPivots(m, rows, cols);

    // Identify free variables
    vector<int> free_vars;
    for (int j = 0; j < cols - 1; j++)
    {
        if (find(pivots.begin(), pivots.end(), j) == pivots.end())
        {
            free_vars.push_back(j);
        }
    }

    // Assign parameters to free variables
    int num_free = free_vars.size();
    vector<string> params;
    for (int i = 0; i < num_free; i++)
    {
        params.push_back("t" + to_string(i + 1));
    }

    // Initialize solution vector with parameters
    vector<double> particular_solution(cols - 1, 0.0);
    vector<vector<double>> basis_vectors;

    // Find a particular solution by setting all free variables to 0
    for (int i = rows - 1; i >= 0; i--)
    {
        // Find the first non-zero coefficient in the row
        int pivot_col = -1;
        for (int j = 0; j < cols - 1; j++)
        {
            if (fabs(m[i][j]) > 1e-12)
            {
                pivot_col = j;
                break;
            }
        }

        if (pivot_col == -1)
        {
            continue; // 0 = 0, skip
        }

        double rhs = m[i][cols - 1];
        for (int j = pivot_col + 1; j < cols - 1; j++)
        {
            rhs -= m[i][j] * particular_solution[j];
        }
        particular_solution[pivot_col] = rhs / m[i][pivot_col];
    }

    // Now, find basis vectors by setting each free variable to 1 and others to 0
    for (int i = 0; i < num_free; i++)
    {
        vector<double> basis(cols - 1, 0.0);
        basis[free_vars[i]] = 1.0; // Set the free variable to 1

        // Perform back-substitution for pivot variables
        for (int r = rank - 1; r >= 0; r--)
        {
            int pivot_col = pivots[r];
            double rhs = 0.0;
            for (int j = pivot_col + 1; j < cols - 1; j++)
            {
                rhs -= m[r][j] * basis[j];
            }
            basis[pivot_col] = rhs / m[r][pivot_col];
        }

        basis_vectors.push_back(basis);
    }

    // Display the general solution
    cout << "General solution:" << endl;
    cout << "x = [";
    for (int j = 0; j < cols - 1; j++)
    {
        cout << fixed << setprecision(4) << particular_solution[j];
        if (j < cols - 2)
            cout << ", ";
    }
    cout << "]";

    for (int i = 0; i < num_free; i++)
    {
        cout << " + " << params[i] << " * [";
        for (int j = 0; j < cols - 1; j++)
        {
            cout << fixed << setprecision(4) << basis_vectors[i][j];
            if (j < cols - 2)
                cout << ", ";
        }
        cout << "]";
        if (i < num_free - 1)
            cout << " + ";
    }
    cout << endl
         << endl;
}

/**
 * @copydoc IdentifyPivots(const std::vector<std::vector<double>> &, int, int)
 */
vector<int> IdentifyPivots(const vector<vector<double>> &m, int rows, int cols)
{
    vector<int> pivots;
    int n = min(rows, cols - 1);
    for (int i = 0; i < n; i++)
    {
        // Find the pivot column in the current row
        int pivot_col = -1;
        for (int j = 0; j < cols - 1; j++)
        {
            if (fabs(m[i][j]) > 1e-12)
            {
                pivot_col = j;
                break;
            }
        }
        if (pivot_col != -1)
            pivots.push_back(pivot_col);
    }
    return pivots;
}
