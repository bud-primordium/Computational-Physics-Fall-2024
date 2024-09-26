/**
 * @author Gilbert Young
 * @date 2024/09/25
 * @file interaction.cpp
 * @brief Implementation of user interaction functions.
 *
 * @details
 * This file implements the functions responsible for interacting with the user, including
 * selecting input files, prompting whether to run the program again, and waiting for the user
 * to exit. These functions guide the flow of the program based on user input.
 */

#include "interaction.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <limits>
#include "utils.h"

using namespace std;

/// Allows the user to select an input `.in` file from the current directory.Returns an empty string if no file is selected.
string SelectInputFile()
{
    vector<string> in_files;
    for (const auto &entry : filesystem::directory_iterator(filesystem::current_path()))
    {
        if (entry.is_regular_file())
        {
            string filename = entry.path().filename().string();
            if (filename.size() >= 3 && filename.substr(filename.size() - 3) == ".in")
            {
                in_files.push_back(filename);
            }
        }
    }

    string selected_file;
    if (in_files.empty())
    {
        cout << "No .in files found in the current directory." << endl;
        return "";
    }
    else if (in_files.size() == 1)
    {
        selected_file = in_files[0];
        cout << "Found one .in file: " << selected_file << ". Automatically selecting it." << endl;
    }
    else
    {
        cout << "Multiple .in files found. Please select one:" << endl;
        for (size_t i = 0; i < in_files.size(); i++)
        {
            cout << i + 1 << ". " << in_files[i] << endl;
        }
        int file_choice;
        // Improved input validation
        while (true)
        {
            cout << "Enter the number of the file you want to use (1-" << in_files.size() << "): ";
            cin >> file_choice;

            if (cin.fail() || file_choice < 1 || file_choice > static_cast<int>(in_files.size()))
            {
                cin.clear();                                         // Clear error flags
                cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear input buffer
                cout << "Invalid input. Please enter a number between 1 and " << in_files.size() << "." << endl;
            }
            else
            {
                break;
            }
        }
        selected_file = in_files[file_choice - 1];
    }
    cout << endl;
    return selected_file;
}

/// Return char The user's choice ('y', 'Y', 'n', 'N').
char AskRunAgain()
{
    char choice;
    while (true)
    {
        cout << "\nDo you want to run the program again? (y/n): ";
        cin >> choice;

        if (choice == 'y' || choice == 'Y' || choice == 'n' || choice == 'N')
        {
            break;
        }
        else
        {
            cout << "Invalid input. Please enter 'y' or 'n'." << endl;
        }
    }
    return choice;
}

/// Waits for the user to press Enter before exiting.
void WaitForExit()
{
    cout << "\nPress Enter to exit...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear input buffer
    cin.get();                                           // Wait for Enter key
}
