#SCI_Lenet-large_client_metrics.csv

def print_csv_column_names(path):
    import pandas as pd
    df = pd.read_csv(path)
    print(f"Columns in {path}: {df.columns.tolist()}")

# Example call for one of your CSV files
print_csv_column_names('SCI_Lenet-large_client_metrics.csv')  # Replace with the actual path to one of your CSV files

