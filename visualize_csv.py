import pandas as pd
import matplotlib.pyplot as plt
import os  # Make sure to import the os module
import sys

def plot_time_metrics(df, output_dir):
    """Create a bar plot for time metrics and save it as an image."""
    time_metrics = df[df['Metric'].str.contains('time', case=False)]
    plt.figure(figsize=(10, 6))
    plt.barh(time_metrics['Metric'], time_metrics['Value'], color='skyblue')
    plt.xlabel('Time (seconds)')
    plt.title('Time Spent on Operations')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_metrics.png")
    plt.close()

def plot_data_metrics(df, output_dir):
    """Create a bar plot for data metrics and save it as an image."""
    data_metrics = df[df['Metric'].str.contains('data sent', case=False)]
    plt.figure(figsize=(10, 6))
    plt.barh(data_metrics['Metric'], data_metrics['Value'], color='lightgreen')
    plt.xlabel('Data Sent (MiB)')
    plt.title('Data Sent in Operations')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_metrics.png")
    plt.close()

def visualize_csv(file_path):
    """Load CSV file and plot metrics."""
    df = pd.read_csv(file_path)
    
    # Convert the Value column to numeric, handling non-numeric values as NaN
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Create output directory if not exists
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the metrics
    plot_time_metrics(df, output_dir)
    plot_data_metrics(df, output_dir)
    print(f"Visualizations saved in the {output_dir} directory.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_csv.py <csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    visualize_csv(csv_file_path)
