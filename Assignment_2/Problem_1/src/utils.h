/*
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: utils.h
@IDE: VSCode
@Formatter: Clang-Format
@Description: Declarations of utility functions for running methods and handling user input.
*/

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <functional>
#include <vector>
#include <map>
#include "methods.h"

extern std::map<std::string, std::vector<RootInfo>> summary;

// Function to run the method and display results
void run_method(const std::string &method_name,
                std::function<long double(long double, long double, long double, int, std::vector<std::string> &, int)> method_func,
                long double a, long double b, long double tol, int max_iter,
                int decimal_places);

// Function to run the problem steps
void run_problem_steps();

// Function to compare all methods
void compare_all_methods();

// Function to get user input
void get_user_input(long double &a, long double &b, long double &x0, std::string &method_name, long double &tol);

// Function to calculate decimal places based on tolerance
int calculate_decimal_places(long double tol);

// Function to run the method and display results (for user-selected methods)
void run_method_user_selection(const std::string &method_name,
                               std::function<long double(long double, long double, long double, int, std::vector<std::string> &, int)> method_func,
                               long double a, long double b, long double tol, int max_iter);

#endif // UTILS_H
