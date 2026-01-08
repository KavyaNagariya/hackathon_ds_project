import pandas as pd

def load_backstories(csv_path):
    """
    Loads the test set CSV into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
        # Standardize column names
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None