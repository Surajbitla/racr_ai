import os
import shutil
import time
from datetime import datetime
from bs4 import BeautifulSoup

# Function to extract the battery capacity value from the battery report
def extract_report_generated_capacity(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Convert all text in the HTML to a list of stripped strings
    text = list(soup.stripped_strings)
    
    # Look for "Report generated" entry and extract capacity value
    for i, line in enumerate(text):
        if 'Report generated' in line:
            for j in range(i, len(text)):
                if 'mWh' in text[j]:
                    return text[j]  # Return the first mWh value after 'Report generated'
    return None

# Function to find the two latest battery report files in the current directory
def find_battery_report_files():
    battery_reports = [f for f in os.listdir() if "battery_report" in f and f.endswith(".html")]
    
    if len(battery_reports) < 2:
        print("Not enough battery reports found.")
        return None, None

    # Sort by modification time to get the most recent two files
    battery_reports.sort(key=os.path.getmtime, reverse=True)
    return battery_reports[0], battery_reports[1]

# Function to calculate time difference between two timestamps
def calculate_time_difference(file1, file2):
    # Extract the timestamps from the filenames
    timestamp_format = "%Y%m%d_%H%M%S"
    time_str1 = file1.split('_')[0] + '_' + file1.split('_')[1]
    time_str2 = file2.split('_')[0] + '_' + file2.split('_')[1]
    
    time1 = datetime.strptime(time_str1, timestamp_format)
    time2 = datetime.strptime(time_str2, timestamp_format)

    # Calculate the difference in seconds
    time_difference = abs((time2 - time1).total_seconds())
    return time_difference

# Function to move files to a destination folder
def move_files_to_folder(files, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # Create folder if it doesn't exist

    for file in files:
        destination = os.path.join(folder_name, file)
        shutil.move(file, destination)
        print(f"Moved {file} to {folder_name}")

# Function to calculate battery consumption by comparing two reports
def calculate_battery_consumption():
    before_file, after_file = find_battery_report_files()
    
    if before_file and after_file:
        before_capacity_str = extract_report_generated_capacity(before_file)
        after_capacity_str = extract_report_generated_capacity(after_file)
        
        if before_capacity_str and after_capacity_str:
            # Convert extracted values to numeric by removing 'mWh' and commas
            before_capacity = int(before_capacity_str.replace('mWh', '').replace(',', '').strip())
            after_capacity = int(after_capacity_str.replace('mWh', '').replace(',', '').strip())
            
            # Show battery capacities
            print(f"Battery after YOLO run: {before_capacity} mWh")
            print(f"Battery before YOLO run: {after_capacity} mWh")
            
            # Calculate the absolute difference
            battery_consumed = abs(after_capacity - before_capacity)
            print(f"Battery consumed: {battery_consumed} mWh")
            
            # Calculate time difference
            time_taken = calculate_time_difference(before_file, after_file)
            print(f"Total time taken: {time_taken} seconds")
            
            # Move the processed files to the "Processed_Reports" folder
            move_files_to_folder([before_file, after_file], "Processed_Reports")
        else:
            print("Failed to extract capacity from one or both reports.")
    else:
        print("Failed to find sufficient battery report files.")

# Run the battery consumption calculation
calculate_battery_consumption()
