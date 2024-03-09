import csv
import random


def generate_random_data(num_records, start_cust_no, existing_coords):
    new_data = []
    for i in range(num_records):
        while True:
            xcoord = random.uniform(0, 50)  # 假设X坐标在0到50之间
            ycoord = random.uniform(0, 50)  # 假设Y坐标在0到50之间
            if (xcoord, ycoord) not in existing_coords:
                # 如果坐标不在现有坐标中，则跳出循环
                break
        cust_no = start_cust_no + i
        profit = random.uniform(100, 500)  # 假设利润在100到500之间
        ready_time = random.randint(0, 100)  # 假设最早时间在0到100之间
        due_time = ready_time + random.randint(50, 300)  # 假设截止时间在最早时间后的50到300时间单位之间
        new_data.append([cust_no, xcoord, ycoord, profit, ready_time, due_time])
        existing_coords.add((xcoord, ycoord))
    return new_data


def append_data_to_tsp_csv(original_filename, new_filename, num_new_records):
    # Read the original data and store the coordinates in a set to check for duplicates
    with open(original_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the header line
        original_data = list(reader)
        existing_coords = set((float(row[1]), float(row[2])) for row in original_data)

    # Find the maximum customer number from the original data
    max_cust_no = max(int(row[0]) for row in original_data)

    # Generate new records without duplicating coordinates
    new_records = generate_random_data(num_new_records, max_cust_no + 1, existing_coords)

    # Combine original and new data
    combined_data = original_data + new_records

    # Write the combined data to the new CSV file
    with open(new_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write the header line
        writer.writerows(combined_data)

    print(f"Stored original and {num_new_records} new records into '{new_filename}' without duplicates.")


# Example usage of the function
append_data_to_tsp_csv('../TSP.csv', 'TSP_large.csv', 100)