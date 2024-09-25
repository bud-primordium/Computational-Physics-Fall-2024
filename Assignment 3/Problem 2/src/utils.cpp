/**
 * @author Gilbert Young
 * @date 2024/09/25
 * @file utils.cpp
 * @brief Implementation of utility functions for matrix operations.
 *
 * @details
 * This file contains the implementations of functions that handle reading matrices from
 * `.in` files and displaying the corresponding system of linear equations. These utility
 * functions are essential for the initialization and output of matrix data used in
 * solving linear systems.
 */

#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

using namespace std;

bool InitMatrix(vector<vector<double>> &m, const string &filename, int &rows, int &cols)
{
    ifstream in(filename);
    if (!in.is_open())
    {
        cerr << "Error: Cannot open file " << filename << endl;
        return false;
    }

    // Read the matrix dimensions dynamically
    string line;
    rows = 0;
    cols = 0;
    vector<vector<double>> temp_matrix;
    while (getline(in, line))
    {
        if (line.empty())
            continue; // Skip empty lines
        vector<double> row;
        double num;
        istringstream iss(line);
        while (iss >> num)
        {
            row.push_back(num);
        }
        if (cols == 0)
        {
            cols = row.size();
        }
        else if ((int)row.size() != cols)
        {
            cerr << "Error: Inconsistent number of columns in the file." << endl;
            in.close();
            return false;
        }
        temp_matrix.push_back(row);
        rows++;
    }
    in.close();

    if (rows == 0 || cols < 2)
    {
        cerr << "Error: The matrix must have at least one equation and one variable." << endl;
        return false;
    }

    // Assign to m
    m = temp_matrix;
    return true;
}

void ShowEquations(const vector<vector<double>> &m, int rows, int cols)
{
    cout << "The current system of linear equations is:" << endl;
    for (int i = 0; i < rows; i++)
    {
        string equation = "";
        for (int j = 0; j < cols - 1; j++)
        {
            // Check if the coefficient is an integer
            double coeff = round(m[i][j] * 1e12) / 1e12; // Handle floating-point precision
            if (fabs(coeff - round(coeff)) < 1e-12)
            {
                equation += to_string(static_cast<long long>(round(coeff))) + "x" + to_string(j + 1);
            }
            else
            {
                equation += to_string(round(m[i][j] * 10000) / 10000.0) + "x" + to_string(j + 1);
            }

            if (j < cols - 2)
                equation += " + ";
        }
        // Handle constant term
        double const_term = round(m[i][cols - 1] * 1e12) / 1e12;
        if (fabs(const_term - round(const_term)) < 1e-12)
        {
            equation += " = " + to_string(static_cast<long long>(round(const_term)));
        }
        else
        {
            equation += " = " + to_string(round(m[i][cols - 1] * 10000) / 10000.0);
        }

        cout << equation << endl;
    }
    cout << endl;
}

bool CheckConsistency(const vector<vector<double>> &m, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        bool all_zero = true;
        for (int j = 0; j < cols - 1; j++)
        {
            if (fabs(m[i][j]) > 1e-12)
            {
                all_zero = false;
                break;
            }
        }
        if (all_zero && fabs(m[i][cols - 1]) > 1e-12)
        {
            return false;
        }
    }
    return true;
}

void DisplaySolution(const vector<double> &solution)
{
    cout << "The system has a unique solution:" << endl;
    for (size_t i = 0; i < solution.size(); i++)
    {
        cout << "x" << i + 1 << " = " << fixed << setprecision(4) << solution[i] << endl;
    }
}
