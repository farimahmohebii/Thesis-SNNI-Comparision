import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_total_time_data(paths, benchmark_name):
    all_data = []

    for snni, path in paths.items():
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"The file {path} is empty.")
            print(f"Columns in {path}: {df.columns.tolist()}")

            # Extract only the total time taken data
            time_data = df[df['Metric'] == 'Total time taken (ms)'].copy()
            time_data['SNNI'] = snni  # Adding the SNNI column
            time_data['Model'] = benchmark_name  # Adding the Model column
            all_data.append(time_data)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    if all_data:
        aggregated_data = pd.concat(all_data, ignore_index=True)
        return aggregated_data
    else:
        print("No data was aggregated.")
        return pd.DataFrame()

def plot_total_time_by_benchmark(benchmark_data, title, output_filename):
    fig, ax = plt.subplots(figsize=(12, 8))

    all_snnis = sorted({snni for data in benchmark_data.values() for snni in data['SNNI'].unique()})
    snni_colors = plt.cm.viridis(np.linspace(0, 1, len(all_snnis)))

    bar_width = 0.2  # Adjust the bar width
    for i, (benchmark, data) in enumerate(benchmark_data.items()):
        for j, snni in enumerate(all_snnis):
            if snni in data['SNNI'].values:
                snni_data = data[data['SNNI'] == snni]
                ax.bar(i + j * bar_width, snni_data['Value'].values[0], color=snni_colors[j], width=bar_width, label=snni if i == 0 else "")

    ax.set_xticks(np.arange(len(benchmark_data)) + bar_width * (len(all_snnis) - 1) / 2)
    ax.set_xticklabels(benchmark_data.keys(), rotation=45)
    ax.set_xlabel('Benchmarks')
    ax.set_ylabel('Total Time Taken (s)')
    
    # Set y-axis to logarithmic scale to make smaller values more visible
    ax.set_yscale('log')
    
    # Add legend outside the plot area
    ax.legend(title='SNNI Approaches', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend

    # Save plot as JPEG
    plt.savefig(output_filename, format='jpeg')
    plt.close()

# Define paths to CSV files for each benchmark in different SNNI approaches

benchmark_paths = {
    'AlexNet': {
        'Cheetah': 'cheetah_alexnet_server_metrics.csv',
        'SCI_HE': 'SCI_HE_alexnet_server_metrics.csv',
        'Porthos Party1': 'porthos_AlexNet_party1_metrics.csv',
        'SCI': 'SCI_AlexNet_server_metrics.csv'
    },
    'DenseNet121': {
        'Cheetah': 'cheetah_densenet121_server_metrics.csv',
        'SCI_HE': 'SCI_HE_densenet121_server_metrics.csv',
        'Porthos Party1': 'porthos_DenseNet_party1_metrics.csv',
        'SCI': 'SCI_DenseNet_server_metrics.csv'
    },
    'LeNet': {
        'Cheetah': 'cheetah_lenet-large_server_metrics.csv',
        'SCI_HE': 'SCI_HE_lenet-large_server_metrics.csv',
        'Porthos Party1': 'porthos_Lenet-large_party1_metrics.csv',
        'SCI': 'SCI_Lenet-large_server_metrics.csv'
    },
    'ResNet50': {
        'Cheetah': 'cheetah_resnet50_server_metrics.csv',
        'SCI_HE': 'SCI_HE_resnet50_server_metrics.csv',
        'Porthos Party1': 'porthos_ResNet_party1_metrics.csv',
        'SCI': 'SCI_ResNet_server_metrics.csv'
    },
    'ShuffleNetV2': {
        'Cheetah': 'cheetah_shufflenetv2_server_metrics.csv',
        'SCI_HE': 'SCI_HE_shufflenetv2_server_metrics.csv',
        'Porthos Party1': 'porthos_ShuffleNetV2_party1_metrics.csv',
        'SCI': 'SCI_ShuffleNetV2_server_metrics.csv'
    },
    'SqueezeNet-ImgNet': {
        'Cheetah': 'cheetah_sqnet_server_metrics.csv',
        'SCI_HE': 'SCI_HE_sqnet_server_metrics.csv',
        'Porthos Party1': 'porthos_SqueezeNetImgNet_party1_metrics.csv',
        'SCI': 'SCI_SqueezeNetImgNet_server_metrics.csv'
    },
    'SqueezeNet-CIFAR10': {
        'Cheetah': 'cheetah_SqueezeNetCIFAR10_server_metrics.csv',
        'SCI_HE': 'SCI_HE_SqueezeNetCIFAR10_server_metrics.csv',
        'Porthos Party1': 'porthos_SqueezeNetCIFAR10_party1_metrics.csv',
        'SCI': 'SCI_SqueezeNetCIFAR10_server_metrics.csv'
    }
}

# Process total time data for each benchmark across different SNNI approaches
benchmark_time_data = {benchmark: process_total_time_data(paths, benchmark) for benchmark, paths in benchmark_paths.items()}

# Plot total time taken for each benchmark across different SNNI approaches
plot_total_time_by_benchmark(benchmark_time_data, 'Total Time Taken Across Benchmarks in Different SNNI Approaches', 'total_time_taken_server.jpeg')
