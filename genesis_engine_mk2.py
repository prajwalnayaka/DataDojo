import json
import random
import csv
import re
from faker import Faker
from pathlib import Path
from datetime import date

fake = Faker()


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
            if re.search(r"binary", rule, flags=re.IGNORECASE):
                fake_row[col_name] = generate_binary()

            elif re.search(r"country", col_name, flags=re.IGNORECASE):
                fake_row[col_name] = fake.country()

            elif re.search(r"date", col_name, flags=re.IGNORECASE):
                oldest_date = date(2021, 1, 1)
                fake_row[col_name] = fake.date_between(
                    start_date=oldest_date, end_date="today"
                )

            elif re.search(r"customerid", col_name, flags=re.IGNORECASE):
                fake_row[col_name] = fake.bothify(text="CUST-####-??")

            elif "description" in col_name.lower() or "name" in col_name.lower():
                fake_row[col_name] = fake.sentence(nb_words=3)[:-1]

            elif re.search(r"charges", col_name, flags=re.IGNORECASE):
                fake_row[col_name] = round(random.uniform(20.0, 8000.0), 2)

            elif re.search(r"categorical", rule, flags=re.IGNORECASE):
                fake_row[col_name] = generate_categorical(rule)

            elif re.search(r"numbers", rule, flags=re.IGNORECASE):
                fake_row[col_name] = generate_number(rule)

            elif re.search(r"text", rule, flags=re.IGNORECASE):
                fake_row[col_name] = f"UID-{random.randint(10000, 99999)}"

            else:
                fake_row[col_name] = "UNKNOWN"

    return fake_row


# ==========================================
# 3. THE PIPELINE (Execution)
# ==========================================


def generate_mk2_dataset(skeletons_folder, output_folder, dataset_index):
    all_files = list(Path(skeletons_folder).glob("*.txt"))
    chosen_file = random.choice(all_files)

    with open(chosen_file, "r") as file:
        full_blueprint = json.load(file)

    # Randomizing Width & Order
    random.shuffle(full_blueprint)

    min_cols = min(5, len(full_blueprint))
    num_cols = random.randint(min_cols, len(full_blueprint))
    final_blueprint = full_blueprint[:num_cols]  # Slices off random columns

    headers = [list(item.keys())[0] for item in final_blueprint]

    num_rows = random.randint(100, 5000)

    output_filename = f"master_key_{dataset_index}.csv"
    output_path = Path(output_folder) / output_filename

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for _ in range(num_rows):
            row_data = generate_row(final_blueprint)
            writer.writerow(row_data)

    print(
        f"Created {output_filename} | Shape: {num_rows} rows x {num_cols} cols | Source: {chosen_file.name}"
    )


# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================

if __name__ == "__main__":
    skeletons_dir = Path(r"D:\DataDojo\Skeletons")
    master_keys_dir = Path(r"D:\DataDojo\Master_Keys")

    master_keys_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        generate_mk2_dataset(skeletons_dir, master_keys_dir, i)
    print(f"🎉 Batch complete! Master keys saved to {master_keys_dir}")
