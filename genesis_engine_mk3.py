import json
import random
import pandas as pd
import csv
import re
from faker import Faker
from pathlib import Path
from datetime import date

fake=Faker()


# ==========================================
# 1. THE LEGO BLOCKS (Generators)
# ==========================================

def generate_number(blueprint_string):
    """
    Parses strings like "Numbers [10:50]" or "Zero & Positive numbers [1-29]"
    and returns a random number in that range.
    """
    # Use regex to find all numbers in the string
    bounds = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", blueprint_string)

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
            if re.search(r"binary",rule, flags=re.IGNORECASE):
                fake_row[col_name] = generate_binary()

            elif re.search(r"country",col_name, flags=re.IGNORECASE):
                fake_row[col_name]=fake.country()

            elif re.search(r"date", col_name, flags=re.IGNORECASE):
                oldest_date = date(2021, 1, 1)
                fake_row[col_name]=fake.date_between(start_date=oldest_date, end_date='today')

            elif re.search(r"customerid",col_name, flags=re.IGNORECASE):
                fake_row[col_name]=fake.bothify(text='CUST-####-??')

            elif "description" in col_name.lower() or "name" in col_name.lower():
                fake_row[col_name] = fake.sentence(nb_words=3)[:-1]

            elif re.search(r"charges",col_name, flags=re.IGNORECASE):
                fake_row[col_name] = round(random.uniform(20.0, 8000.0), 2)

            elif re.search(r"categorical",rule, flags=re.IGNORECASE):
                fake_row[col_name] = generate_categorical(rule)

            elif re.search(r"numbers",rule, flags=re.IGNORECASE):
                fake_row[col_name] = generate_number(rule)

            elif re.search(r"text",rule, flags=re.IGNORECASE):
                fake_row[col_name] = f"UID-{random.randint(10000, 99999)}"

            else:
                fake_row[col_name] = "UNKNOWN"


    return fake_row


# ==========================================
# 3. THE PIPELINE (Execution)
# ==========================================

def run_genesis(blueprint_path, num_rows):
    dataset_memory = []
    # 1. Load the blueprint txt/json file
    with open(blueprint_path, 'r') as file:
        blueprint = json.load(file)

    # 2. Extract column headers for the CSV
    headers = []
    for item in blueprint:
        headers.extend(item.keys())
    dataset_memory.append(headers)

    for _ in range(num_rows):
        row_data = generate_row(blueprint)
        dataset_memory.append(row_data)

    print(f"✅ Genesis Complete: {num_rows}.")

    return pd.DataFrame(dataset_memory)


folder_path = Path(r"D:\DataDojo\Skeletons")
output_dir = Path(r"D:\DataDojo\Generated Datasets")

output_dir.mkdir(parents=True, exist_ok=True)

skeletons=list(folder_path.glob("*.txt"))
skeleton=random.choice(skeletons)
first_4=skeleton[:4]

output_filename = f"generated_dataset_{first_4}.csv"
output_csv_path = output_dir / output_filename
print(f"⚙️ Processing blueprint {skeleton}...")
skeleton.append(".txt")
num_rows = random.randint(100, 5000)

run_genesis(skeleton, num_rows=num_rows)