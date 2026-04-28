import pandas as pd

def get_numerical_and_categorical_features(df: pd.DataFrame) -> tuple[list, list]:
    num_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = df.select_dtypes(include=["object"]).columns.tolist()
    return num_features, cat_features


