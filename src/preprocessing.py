import pandas as pd

def get_numerical_and_categorical_features(
    df: pd.DataFrame,
    ignore_columns: list = None
) -> tuple[list, list]:
    """
    Identify numerical and categorical features in a DataFrame.

    Args:
        df: Pandas DataFrame to analyze.
        ignore_columns: List of column names to exclude from the analysis.

    Returns:
        tuple: (numerical_features, categorical_features)
    """
    if ignore_columns is None:
        ignore_columns = []
    
    # Get all numerical and categorical features
    num_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Filter out ignored columns
    num_features = [col for col in num_features if col not in ignore_columns]
    cat_features = [col for col in cat_features if col not in ignore_columns]
    
    return num_features, cat_features
