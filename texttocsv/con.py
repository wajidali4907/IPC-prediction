import pandas as pd

# Read the text data from your file (replace 'your_file.txt' with your actual file path)
with open('/home/wajid/IPC prediction/csv/stats.txt', 'r') as file:
    lines = file.readlines()

# Initialize a list to store dictionaries representing rows of data
data_rows = []

# Flag to indicate when to start parsing data
start_parsing = False

# Initialize a list to store column names
column_names = []

# Iterate through the lines
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace
    if line.startswith('---------- Begin Simulation Statistics ----------'):
        start_parsing = True
        current_data = {}  # Initialize a dictionary to store data for this section
    elif start_parsing and line:  # Check for non-empty lines when parsing
        # Split each line into key and value using the first occurrence of whitespace
        key, value = line.split(None, 1)
        current_data[key] = value.strip()  # Remove leading/trailing whitespace
        # Append the key as a column name
        column_names.append(key)
    elif start_parsing and not line:  # Blank line indicates the end of a section
        start_parsing = False
        data_rows.append(current_data)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_rows)

# Save the DataFrame as a CSV file (replace 'output.csv' with your desired file name)
df.to_csv('/home/wajid/IPC prediction/csv/output.csv', index=False)
