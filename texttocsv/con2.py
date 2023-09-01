import os
import re
import csv

# Directory containing your text files
directory = 'stats'

# Define the metric names you want to extract
metric_names = [
    'system.o3Cpu0.cpi',
    'system.o3Cpu1.cpi',
    'system.o3Cpu2.cpi',
    'system.o3Cpu3.cpi',
    'system.o3Cpu4.cpi',
    'system.o3Cpu5.cpi',
    'system.o3Cpu6.cpi',
    'system.o3Cpu7.cpi',
    'system.o3Cpu0.numLoadInsts',
    'system.o3Cpu1.numLoadInsts',
    'system.o3Cpu2.numLoadInsts',
    'system.o3Cpu3.numLoadInsts',
    'system.o3Cpu4.numLoadInsts',
    'system.o3Cpu5.numLoadInsts',
    'system.o3Cpu6.numLoadInsts',
    'system.o3Cpu7.numLoadInsts',
    'system.o3Cpu0.numStoreInsts',
    'system.o3Cpu1.numStoreInsts',
    'system.o3Cpu2.numStoreInsts',
    'system.o3Cpu3.numStoreInsts',
    'system.o3Cpu4.numStoreInsts',
    'system.o3Cpu5.numStoreInsts',
    'system.o3Cpu6.numStoreInsts',
    'system.o3Cpu7.numStoreInsts',
    'system.o3Cpu0.numInsts',
    'system.o3Cpu1.numInsts',
    'system.o3Cpu2.numInsts',
    'system.o3Cpu3.numInsts',
    'system.o3Cpu4.numInsts',
    'system.o3Cpu5.numInsts',
    'system.o3Cpu6.numInsts',
    'system.o3Cpu7.numInsts',
    'system.o3Cpu0.numBranches',
    'system.o3Cpu1.numBranches',
    'system.o3Cpu2.numBranches',
    'system.o3Cpu3.numBranches',
    'system.o3Cpu4.numBranches',
    'system.o3Cpu5.numBranches',
    'system.o3Cpu6.numBranches',
    'system.o3Cpu7.numBranches',
    'system.o3Cpu0.intAluAccesses',
    'system.o3Cpu1.intAluAccesses',
    'system.o3Cpu2.intAluAccesses',
    'system.o3Cpu3.intAluAccesses',
    'system.o3Cpu4.intAluAccesses',
    'system.o3Cpu5.intAluAccesses',
    'system.o3Cpu6.intAluAccesses',
    'system.o3Cpu7.intAluAccesses',
    'system.o3Cpu0.thread_0.numOps',
    'system.o3Cpu1.thread_0.numOps',
    'system.o3Cpu2.thread_0.numOps',
    'system.o3Cpu3.thread_0.numOps',
    'system.o3Cpu4.thread_0.numOps',
    'system.o3Cpu5.thread_0.numOps',
    'system.o3Cpu6.thread_0.numOps',
    'system.o3Cpu7.thread_0.numOps',
    'system.o3Cpu0.ipc',
    'system.o3Cpu1.ipc',
    'system.o3Cpu2.ipc',
    'system.o3Cpu3.ipc',
    'system.o3Cpu4.ipc',
    'system.o3Cpu5.ipc',
    'system.o3Cpu6.ipc',
    'system.o3Cpu7.ipc'
]

# Initialize a list to store the extracted data
all_data = []

# Loop through the text files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        
        # Initialize a dictionary to store metrics from each file
        file_data = {}
        
        # Read the text file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Parse the data and extract the desired metrics
        for metric_name in metric_names:
            file_data[metric_name] = 0  # Initialize with 0

            for line in lines:
                if metric_name in line:
                    match = re.search(r'([\d.]+)\s*#', line)
                    if match:
                        value = float(match.group(1))
                        file_data[metric_name] = value
        
        # Append the data from this file to the list
        all_data.append(file_data)

# Define the output CSV file name
output_csv = 'output1.csv'

# Write the extracted data to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=metric_names)
    writer.writeheader()
    writer.writerows(all_data)

print(f'Extracted data from {len(all_data)} text files and saved to {output_csv}')
