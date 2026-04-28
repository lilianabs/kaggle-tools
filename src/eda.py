import pandas as pd

def compute_percentage_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    missing_values = pd.DataFrame(df.isna().sum() / df.shape[0])

    missing_values.reset_index(inplace=True)
    missing_values = missing_values.rename(columns = {'index':'feature',
                                                                     0: 'prct_missing'})
 
    missing_values = missing_values.sort_values(by='prct_missing', ascending=False)
    return missing_values

