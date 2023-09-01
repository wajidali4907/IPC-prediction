import pandas as pd

# Path to your input text file (use absolute path)
input_file = '/home/wajid/IPC prediction/csv/2048.txt'

# Path to the output CSV file (use absolute path)
output_csv = '/home/wajid/IPC prediction/csv/2048.csv'

# Initialize empty lists to store data
labels = []
values = []

# Open the input file and read line by line
with open(input_file, 'r') as file:
    for line in file:
        # Split each line into label and value using whitespace as the delimiter
        parts = line.strip().split()
        if len(parts) >= 2:
            label = parts[0]
            value = parts[1]
            # Append the label and value to the respective lists
            labels.append(label)
            values.append(value)

# Create a DataFrame from the lists
data = pd.DataFrame({'Parameter': labels, 'Value': values})

# Save the DataFrame to a CSV file
data.to_csv(output_csv, index=False)

print("CSV file saved successfully.")
