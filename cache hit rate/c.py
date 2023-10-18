import os
import re
import csv

# Directory containing your text files
directory = '/home/wajid/IPC prediction/stats'

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
    'system.o3Cpu7.ipc',
    'system.caches.controllers0.L1Dcache.m_demand_hits',
    'system.caches.controllers1.L1Dcache.m_demand_hits',
    'system.caches.controllers2.L1Dcache.m_demand_hits',
    'system.caches.controllers3.L1Dcache.m_demand_hits',
    'system.caches.controllers4.L1Dcache.m_demand_hits',
    'system.caches.controllers5.L1Dcache.m_demand_hits',
    'system.caches.controllers6.L1Dcache.m_demand_hits',
    'system.caches.controllers7.L1Dcache.m_demand_hits'
]

# Initialize dictionaries to store the extracted data
aggregated_data = {metric.split('.')[-1]: [] for metric in metric_names}
model_cache_mapping = {}  # To map cache controller metrics to CPU metrics

# Loop through the text files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)

        # Determine the model number and cache values based on the file name pattern
        if 'aggressive' in filename:
            model_number = 1.0
            L1D_cache = 1048576
            L1I_cache = 65536
            L2_cache = 2097152
            pipeline_width = 16
        elif 'base' in filename:
            model_number = 2.0
            L1D_cache = 524288
            L1I_cache = 32768
            L2_cache = 1048576
            pipeline_width = 8
        elif 'lean' in filename:
            model_number = 3.0
            L1D_cache = 262144
            L1I_cache = 16384
            L2_cache = 524288
            pipeline_width = 4
        else:
            model_number = 0.0  # Default model number if none of the keywords are found

        # Read the text file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Parse the data and extract the desired metrics
        metric_values = {metric: [] for metric in metric_names}

        # Extract metrics for CPU cores and cache controllers
        for line in lines:
            for metric_name in metric_names:
                # Check if the metric corresponds to a CPU core or cache controller
                if metric_name in line:
                    match = re.search(r'([\d.]+)\s*#', line)
                    if match:
                        value = float(match.group(1))
                        metric_values[metric_name].append(value)

        # Append the aggregated metric values
        for metric_name in metric_names:
            aggregated_data[metric_name.split('.')[-1]].extend(metric_values[metric_name])

# Define the output CSV file name
output_csv = 'output.csv'

# Write the extracted data to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    header_row = ['model'] + list(aggregated_data.keys()) + ['L1Icache', 'L1Dcache', 'L2cache', 'pipelinewidth']
    writer.writerow(header_row)

    # Calculate the number of samples
    num_samples = len(aggregated_data[list(aggregated_data.keys())[0]])
    print(f"Number of samples: {num_samples}")

    for i in range(num_samples):
        row = [
            model_number,
            *[aggregated_data[metric][i] if aggregated_data[metric] else None for metric in aggregated_data.keys()],
            L1I_cache,
            L1D_cache,
            L2_cache,
            pipeline_width
        ]
        writer.writerow(row)

print(f'Extracted data from text files and saved to {output_csv}')
