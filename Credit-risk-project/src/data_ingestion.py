import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the German Credit dataset from CSV.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: dataset as pandas dataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    data_path = "data/german_credit_data.csv"
    df = load_data(data_path)
    print(df.head())
    print(f"Dataset shape: {df.shape}")
    
    