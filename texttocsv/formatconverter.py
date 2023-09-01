import csv

# Open the text file in read mode
with open('input.txt', 'r') as text_file:
    # Process the text data (for example, split by whitespace)
    lines = text_file.readlines()
    processed_data = [line.strip().split() for line in lines]

# Define the name for the CSV file
csv_file_name = 'output.csv'

# Write the processed data to a CSV file
with open(csv_file_name, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header row (if needed)
    # csv_writer.writerow(['Column1', 'Column2', ...])

    # Write the data rows
    csv_writer.writerows(processed_data)

print(f'Text file "{text_file.name}" converted to CSV file "{csv_file_name}"')
