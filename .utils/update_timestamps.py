"""
@file update_timestamps.py
@brief This script automatically updates the "@date" tag in the header comments of all code files in the current directory and its subdirectories.
 
It scans supported file types (.py, .h, .cpp, .f90) and modifies the first five lines of each file, updating the date based on the current system time.

@author Gilbert Young
@date 2024/09/24
@details
This task is used for batch updating the date in a project's code files. The script searches for the "@date" tag and replaces it with the current date.

- IDE: VSCode
- Formatter: Black
"""

import os
import sys
import re
from datetime import datetime


def update_date(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        updated = False
        # Regular expression to match @date tag
        date_pattern = re.compile(r"@date\s*\d{4}/\d{2}/\d{2}")

        # Only check and update the date in the first five lines
        for i in range(min(5, len(lines))):
            if date_pattern.search(lines[i]):
                new_date = datetime.now().strftime("%Y/%m/%d")
                lines[i] = date_pattern.sub(f"@date {new_date}", lines[i])
                updated = True

        if updated:
            with open(file_path, "w", encoding="utf-8") as file:
                file.writelines(lines)

    except Exception as e:
        print(f"Failed to update {file_path}: {e}")


def update_all_files(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith((".py", ".h", ".cpp", ".f90")):
                update_date(os.path.join(root, filename))


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    update_all_files(directory)
