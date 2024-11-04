import subprocess
import time
import os
from datetime import datetime

# Function to generate and rename the battery report
def generate_battery_report(prefix="before"):
    # Set the output path explicitly
    output_dir = r"C:\Users\GenCyber\Documents\RACR_AI_New"
    output_file = os.path.join(output_dir, "battery-report.html")

    # Generate the battery report and save it to the specified path
    subprocess.run(f"powercfg /batteryreport /output {output_file}", shell=True)

    # Generate the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define destination file path with a unique name
    dest_file = os.path.join(output_dir, f"{prefix}_battery_report_{timestamp}.html")
    
    # Rename the battery report to avoid overwriting
    if os.path.exists(output_file):
        os.rename(output_file, dest_file)
        print(f"Battery report saved as {dest_file}")
    else:
        print(f"Battery report not found at {output_file}")

# Step 1: Generate the battery report before running the YOLO code
generate_battery_report(prefix="before")

# Step 2: Run the entire YOLO processing code
print("Running YOLO code...")
time.sleep(5)  # Simulating the YOLO code execution

# Step 3: Generate the battery report after the YOLO code finishes
generate_battery_report(prefix="after")

print("Battery reports generated before and after YOLO processing.")

# Step 4: Call the mwh_extractor.py script using subprocess
try:
    # Run the mwh_extractor.py script after YOLO processing completes
    subprocess.run(['python', 'mwh_extractor.py'], check=True)
    print("mwh_extractor.py executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running mwh_extractor.py: {e}")
