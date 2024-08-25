import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_benchmark_data(files):
    """Load benchmark data from the given CSV files."""
    benchmark_data = []

    for file in files:
        df = pd.read_csv(file)
        df['Benchmark'] = os.path.splitext(os.path.basename(file))[0]  # Add a column for benchmark name
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')  # Ensure numeric values
        benchmark_data.append(df)

    # Concatenate all benchmark data into a single DataFrame
    all_data = pd.concat(benchmark_data, ignore_index=True)
    return all_data

def plot_comparison(all_data, output_dir):
    """Plot comparison charts for time and data metrics and save as images."""
    # Pivot the data for comparison
    pivot_time = all_data.pivot(index='Metric', columns='Benchmark', values='Value')
    pivot_time = pivot_time.filter(like='time', axis=0)

    # Plot comparison of time metrics across benchmarks
    pivot_time.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'orange', 'lightgreen'])
    plt.title('Comparison of Time Metrics Across Benchmarks')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    plt.close()

    # Plot comparison of data metrics across benchmarks with a logarithmic scale
    pivot_data = all_data.pivot(index='Metric', columns='Benchmark', values='Value')
    pivot_data = pivot_data.filter(like='data sent', axis=0)

    pivot_data.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'orange', 'lightgreen'])
    plt.title('Comparison of Data Sent Across Benchmarks')
    plt.ylabel('Data Sent (MiB)')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'data_comparison_log.png'))
    plt.close()

    # Heatmap for time metrics
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_time, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.title('Heatmap of Time Metrics Across Benchmarks')
    plt.ylabel('Metrics')
    plt.xlabel('Benchmarks')
    plt.savefig(os.path.join(output_dir, 'time_heatmap.png'))
    plt.close()

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Compare benchmark results from CSV files.')
    parser.add_argument('csv_files', nargs='+', help='CSV files to compare (2 to 4 files).')

    # Parse arguments
    args = parser.parse_args()

    # Load and process benchmark data
    all_data = load_benchmark_data(args.csv_files)

    # Create output directory if it doesn't exist
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the comparison charts and save as images
    plot_comparison(all_data, output_dir)
    print(f"Plots saved to {output_dir} directory.")

if __name__ == "__main__":
    main()
