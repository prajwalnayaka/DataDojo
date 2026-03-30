import json
import random
import csv
import re
from pathlib import Path


# ==========================================
# 1. THE LEGO BLOCKS (Generators)
# ==========================================

def generate_number(blueprint_string):
    """
    Parses strings like "Numbers [10:50]" or "Zero & Positive numbers [1-29]"
    and returns a random number in that range.
    """
    # Use regex to find all numbers in the string
    bounds = re.findall(r"[-+]?\d*\.\d+|\d+", blueprint_string)

    if len(bounds) >= 2:
        min_val = float(bounds[0])
        max_val = float(bounds[1])

        # If both are whole numbers, return an integer. Otherwise, float.
        if min_val.is_integer() and max_val.is_integer():
            return random.randint(int(min_val), int(max_val))
        else:
            return round(random.uniform(min_val, max_val), 2)
    return 0


def generate_categorical(cardinality_type):
    """
    Returns a random fake string based on whether it's High or Low cardinality.
    """
    low_options = ["Option A", "Option B", "Option C"]
    high_options = [f"Type_{i}" for i in range(1, 100)]

    if "Low" in cardinality_type:
        return random.choice(low_options)
    else:
        return random.choice(high_options)


def generate_binary():
    """Returns Yes or No."""
    return random.choice(["Yes", "No"])


# ==========================================
# 2. THE ENGINE (Row Builder)
# ==========================================

def generate_row(blueprint):
    """
    Takes the loaded JSON blueprint and generates one single dictionary
    representing a fake row of data.
    """
    fake_row = {}

    for column_dict in blueprint:
        # blueprint is a list of dicts: [{"Age": "Zero & Positive numbers [18-60]"}]
        for col_name, rule in column_dict.items():

            if "Binary" in rule:
                fake_row[col_name] = generate_binary()

            elif "Categorical" in rule:
                fake_row[col_name] = generate_categorical(rule)

            elif "Numbers" in rule or "integers" in rule:

                fake_row[col_name] = generate_number(rule)
            elif "Text/String" in rule:
                fake_row[col_name] = f"UID-{random.randint(10000, 99999)}"

            else:
                fake_row[col_name] = "UNKNOWN"

    return fake_row


# ==========================================
# 3. THE PIPELINE (Execution)
# ==========================================

def run_genesis(blueprint_path, output_csv_path, num_rows=250):
    # 1. Load the blueprint txt/json file
    with open(blueprint_path, 'r') as file:
        blueprint = json.load(file)

    # 2. Extract column headers for the CSV
    headers = []
    for item in blueprint:
        headers.extend(item.keys())

    # 3. Generate data and write to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for _ in range(num_rows):
            row_data = generate_row(blueprint)
            writer.writerow(row_data)

    print(f"✅ Genesis Complete: {num_rows} rows saved to {output_csv_path}")

# --- TEST IT ---
# Point this to one of your 8 .txt files!
# run_genesis(r"D:\DataDojo\Human_Blueprint.txt", "fake_human_data.csv", num_rows=50)

folder_path = Path(r"D:\DataDojo\Skeletons")
output_dir = Path(r"D:\DataDojo\Generated Datasets")

output_dir.mkdir(parents=True, exist_ok=True)

for index, file in enumerate(folder_path.glob("*.txt"), start=1):
    output_filename = f"generated_dataset_{index}.csv"
    output_csv_path = output_dir / output_filename
    print(f"⚙️ Processing blueprint {index}: {file.name}...")
    run_genesis(file, output_csv_path, num_rows=250)