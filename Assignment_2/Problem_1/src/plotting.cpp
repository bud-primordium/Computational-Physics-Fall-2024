/*
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: plotting.cpp
@IDE: VSCode
@Formatter: Clang-Format
@Description: Definition of function to plot f(x) on a grid.
*/

#include "plotting.h"
#include "functions.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <sstream>

// Function to plot f(x) on a grid
void plot_function(long double x_min, long double x_max, long double y_min, long double y_max,
                   int width, int height, long double label_interval)
{
    std::vector<std::string> grid(height, std::string(width, ' '));
    int x_axis = -1, y_axis = -1;

    // Determine x-axis position
    if (y_min <= 0 && y_max >= 0)
    {
        x_axis = static_cast<int>(round((0 - y_min) / (y_max - y_min) * (height - 1)));
    }

    // Determine y-axis position
    if (x_min <= 0 && x_max >= 0)
    {
        y_axis = static_cast<int>(round((0 - x_min) / (x_max - x_min) * (width - 1)));
    }

    // Plot the function
    for (int i = 0; i < width; ++i)
    {
        long double x = x_min + i * (x_max - x_min) / (width - 1);
        long double y = f(x);
        if (y < y_min || y > y_max)
            continue;
        int j = static_cast<int>(round((y - y_min) / (y_max - y_min) * (height - 1)));
        if (j >= 0 && j < height)
        {
            grid[height - 1 - j][i] = '*';
        }
    }

    // Draw x-axis
    if (x_axis != -1)
    {
        for (int i = 0; i < width; ++i)
        {
            if (grid[x_axis][i] == ' ')
                grid[x_axis][i] = '-';
        }
    }

    // Draw y-axis
    if (y_axis != -1)
    {
        for (int i = 0; i < height; ++i)
        {
            if (grid[i][y_axis] == ' ')
                grid[i][y_axis] = '|';
        }
    }

    // Draw origin
    if (x_axis != -1 && y_axis != -1)
    {
        grid[x_axis][y_axis] = '+';
    }

    // Print the grid
    std::cout << "\nFunction Plot:\n";
    for (const auto &row : grid)
    {
        std::cout << row << '\n';
    }

    // Print x-axis labels
    std::string label_line(width, ' ');
    for (int label = static_cast<int>(ceil(x_min / label_interval)) * static_cast<int>(label_interval);
         label <= static_cast<int>(floor(x_max / label_interval)) * static_cast<int>(label_interval);
         label += static_cast<int>(label_interval))
    {
        double relative_pos = (static_cast<double>(label) - x_min) / (x_max - x_min);
        int pos = static_cast<int>(round(relative_pos * (width - 1)));
        std::ostringstream oss_label;
        oss_label << std::fixed << std::setprecision(0) << label;
        std::string label_str = oss_label.str();
        int start_pos = pos - static_cast<int>(label_str.length() / 2);
        if (start_pos < 0)
            start_pos = 0;
        if (start_pos + static_cast<int>(label_str.length()) > width)
            continue;
        for (size_t i = 0; i < label_str.length(); ++i)
        {
            label_line[start_pos + i] = label_str[i];
        }
    }
    std::cout << label_line << std::endl;
    std::cout << "x range: [" << x_min << ", " << x_max << "]\n";
    std::cout << "y range: [" << y_min << ", " << y_max << "]\n\n";
}
