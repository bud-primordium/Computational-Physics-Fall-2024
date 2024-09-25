/**
 * @author Gilbert Young
 * @date 2024/09/25
 * @file interaction.h
 * @brief User interaction functions.
 */

#ifndef INTERACTION_H
#define INTERACTION_H

#include <string>

/**
 * @brief Allows the user to select an input .in file from the current directory.
 *
 * @return std::string The name of the selected file. Empty string if no file is selected.
 */
std::string SelectInputFile();

/**
 * @brief Asks the user if they want to run the program again.
 *
 * @return char The user's choice ('y', 'Y', 'n', 'N').task 
 */
char AskRunAgain();

/**
 * @brief Waits for the user to press Enter before exiting.
 */
void WaitForExit();

#endif // INTERACTION_H
