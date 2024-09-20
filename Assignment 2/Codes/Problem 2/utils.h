/*
@Author: Gilbert Young
@Time: 2024/09/19 08:56
@File_name: utils.h
@Description:
Header file containing utility functions:
1. displayDefaultParameters: displays the default parameters for the selected algorithm.
2. compareMethods: compares all optimization methods using default parameters.
*/

#ifndef UTILS_H
#define UTILS_H

#include "structs.h"

void displayDefaultParameters(const DefaultParameters &params, int option);
void compareMethods(const DefaultParameters &params);

#endif // UTILS_H
