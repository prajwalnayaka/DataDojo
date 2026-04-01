import pandas as pd
import numpy as np
import random
from pathlib import Path
from genesis_engine_mk3 import generate_mk3_dataframe

# ==========================================
# 1. THE SABOTAGE FUNCTIONS
# ==========================================


def sabotage_level_1(df):
    """Injects exact duplicates and missing values (NaNs)."""
    dirty_df = df.copy()

    # Sabotage A: Duplicate 5% of the rows and shuffle them
    num_dupes = max(1, int(len(dirty_df) * 0.05))
    dupes = dirty_df.sample(n=num_dupes, replace=True)
    dirty_df = pd.concat([dirty_df, dupes], ignore_index=True)
    dirty_df = dirty_df.sample(frac=1.0).reset_index(drop=True)  # Shuffle

    # Sabotage B: Punch holes (NaNs) into 30% of a random column
    target_col_holes = random.choice(dirty_df.columns)
    mask = np.random.rand(len(dirty_df)) < 0.3  # Some pandas bs magic
    dirty_df.loc[mask, target_col_holes] = np.nan

    # Sabotage C: Completely empty a random column (all NaNs)
    available_cols = [col for col in dirty_df.columns if col != target_col_holes]
    if available_cols:
        target_col_empty = random.choice(available_cols)
        dirty_df[target_col_empty] = np.nan

    return dirty_df


def sabotage_level_2(df):
    """Corrupts numeric columns with strings and symbols."""
    dirty_df = sabotage_level_1(df)  # Level 2 includes Level 1 messes

    numeric_cols = dirty_df.select_dtypes(include=["number"]).columns.tolist()
    target_col = ""

    if numeric_cols:
        target_col = random.choice(numeric_cols)
        # Sabotage: Convert perfectly good floats into nasty currency strings (e.g., 4500.5 -> "$4,500.50")
        dirty_df[target_col] = dirty_df[target_col].apply(
            lambda x: (
                random.choice(
                    [
                        f"${x:,.2f}" if pd.notnull(x) else x,
                        f"{x:.2f}." if pd.notnull(x) else x,
                    ]
                )
                if pd.notnull(x)
                else x
            )
        )

    return target_col, dirty_df


def sabotage_level_3(df):
    """Injects text inconsistencies."""
    level_2_target_col, dirty_df = sabotage_level_2(
        df
    )  # Level 3 includes Level 1 & 2 messes

    # Find text/categorical columns
    text_cols = dirty_df.select_dtypes(include=["object"]).columns.tolist()

    if text_cols:
        for target_col in text_cols:
            # Sabotage: Randomly change the casing of the text to break groupbys and value_counts
            if random.random() < 0.3 and target_col != level_2_target_col:
                dirty_df[target_col] = dirty_df[target_col].apply(
                    lambda x: (
                        str(x).upper()
                        if random.random() > 0.7
                        else (str(x).lower() if random.random() < 0.3 else x)
                    )
                )

    return dirty_df


# ==========================================
# 2. THE CORRUPTION PIPELINE
# ==========================================


def run_ruiner(master_df,difficulty):

    if difficulty == 'Easy':
        easy_dirty_df = sabotage_level_1(master_df.copy())
        return easy_dirty_df

    elif difficulty == 'Medium':
        _, medium_dirty_df = sabotage_level_2(master_df.copy())
        return medium_dirty_df

    else:
        hard_dirty_df = sabotage_level_3(master_df.copy())
        return hard_dirty_df


if __name__ == "__main__":
    skeletons_dir = Path(r"D:\DataDojo\Skeletons")
    master_df = generate_mk3_dataframe(skeletons_dir)

    dirty_dataset = run_ruiner(master_df, difficulty='Easy')
