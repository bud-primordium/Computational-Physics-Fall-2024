/*
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: methods.h
@IDE: VSCode
@Formatter: Clang-Format
@Description: Declaration of various root-finding methods.
*/

#ifndef METHODS_H
#define METHODS_H

#include <vector>
#include <string>
#include "functions.h"

struct RootInfo
{
    long double root;   // Root value
    int iterations;     // Number of iterations
    int decimal_places; // Number of decimal places to display
};

// Bisection Method
long double bisection(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places);

// Newton-Raphson Method
long double newton_raphson(long double x0, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places);

// Hybrid Method (Bisection + Newton-Raphson)
long double hybrid_method(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places);

// Brent's Method
long double brent_method(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places);

// Ridder's Method
long double ridder_method(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places);

#endif // METHODS_H
