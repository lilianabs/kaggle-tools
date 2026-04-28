import pandas as pd

def get_column_groups(df: pd.DataFrame) -> tuple[list, list]:
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return num_cols, cat_cols

