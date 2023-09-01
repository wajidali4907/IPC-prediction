import os
import re
import csv

# Directory containing your text files
directory = 'stats'

# Define the metric names you want to extract
metric_names = [
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

# Initialize a dictionary to store the extracted data
aggregated_data = {metric.split('.')[-1]: [] for metric in metric_names}

# Loop through the text files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)

        # Read the text file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Parse the data and extract the desired metrics for each CPU core
        for metric_name in metric_names:
            metric_values = []

            for core_id in range(8):
                for line in lines:
                    if f'system.o3Cpu{core_id}.' in line and metric_name in line:
                        match = re.search(r'([\d.]+)\s*#', line)
                        if match:
                            value = float(match.group(1))
                            metric_values.append(value)

            # Append the aggregated metric values to the corresponding column
            aggregated_data[metric_name.split('.')[-1]].extend(metric_values)

# Define the output CSV file name
output_csv = 'output.csv'

# Write the extracted data to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(aggregated_data.keys())  # Write the header row
    # Transpose the data and write it to the CSV
    num_samples = len(aggregated_data[list(aggregated_data.keys())[0]])
    for i in range(num_samples):
        row = [aggregated_data[metric][i] for metric in aggregated_data.keys()]
        writer.writerow(row)

print(f'Extracted data from text files and saved to {output_csv}')
