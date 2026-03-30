import pandas as pd
from pathlib import Path
import json

def extract_skeleton(path):
    path = Path(path)
    skeleton = []
    binary = ["yes", "no"]
    datatypes=["int64","float64"]
    if path.exists() and path.is_file():
        ds=pd.read_csv(path)
        for column in ds.columns:
            if set(ds[column].dropna().astype(str).str.lower().unique()) == set(binary):
                skeleton.append({column:"Binary yes/no"})

            elif ds[column].nunique()<=5:
                if ds[column].unique().dtype == "object":
                    skeleton.append({column:"Categorical. Low cardinality"})
                elif ds[column].unique().dtype in datatypes and (ds[column] >= 0).all():
                    min = ds[column].unique().min()
                    max = ds[column].unique().max()
                    skeleton.append({column: f"Zero & Positive numbers. Low cardinality. [{min}:{max}]"})

            elif 5 < ds[column].nunique() <= 10:
                if ds[column].dropna().unique().dtype=="object":
                    skeleton.append({column:"Categorical. High cardinality"})
                elif ds[column].unique().dtype in datatypes and (ds[column] >= 0).all():
                    min = ds[column].unique().min()
                    max = ds[column].unique().max()
                    skeleton.append({column: f"Zero & Positive numbers. High cardinality. [{min}:{max}]"})

            elif ds[column].unique().dtype in datatypes:
                #if (ds[column] >= 0).all():
                    min=ds[column].unique().min()
                    max=ds[column].unique().max()
                    skeleton.append({column:f"Numbers [{min}:{max}]"})

            elif ds[column].dtype == "object":
                skeleton.append({column: "Text/String. Very High Cardinality"})

            else:
                skeleton.append({column: f"Unhandled Type: {ds[column].dtype}"})

    print(json.dumps(skeleton,indent=4))

extract_skeleton(r"D:\DataDojo\Services\CustomerChurn.csv")