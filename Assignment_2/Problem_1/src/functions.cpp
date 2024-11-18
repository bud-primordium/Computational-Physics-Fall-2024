/*
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: functions.cpp
@IDE: VSCode
@Formatter: Clang-Format
@Description: Definition of the function f(x) = x^3 - 5x + 3 and its derivative f'(x) = 3x^2 - 5.
*/

#include "functions.h"

// Function f(x) = x^3 - 5x + 3
long double f(long double x)
{
    return x * x * x - 5 * x + 3;
}

// Derivative f'(x) = 3x^2 - 5
long double f_prime(long double x)
{
    return 3 * x * x - 5;
}
