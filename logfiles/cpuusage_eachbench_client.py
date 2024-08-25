import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_cpu_data_by_benchmark(paths, benchmark_name):
    all_data = []

    for snni, path in paths.items():
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"The file {path} is empty.")
            print(f"Columns in {path}: {df.columns.tolist()}")

            # Extract only the average CPU usage data
            cpu_data = df[df['Metric'] == 'Average CPU usage (%)'].copy()
            cpu_data['SNNI'] = snni  # Adding the SNNI column
            cpu_data['Model'] = benchmark_name  # Adding the Model column
            all_data.append(cpu_data)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    if all_data:
        aggregated_data = pd.concat(all_data, ignore_index=True)
        return aggregated_data
    else:
        print("No data was aggregated.")
        return pd.DataFrame()

def plot_cpu_usage_by_benchmark(benchmark_data, title, output_filename):
    num_benchmarks = len(benchmark_data)
    fig, axs = plt.subplots((num_benchmarks + 2) // 3, 3, figsize=(18, 10))  # Adjusted grid size based on number of benchmarks

    snni_colors = plt.cm.viridis(np.linspace(0, 1, len(benchmark_data[next(iter(benchmark_data))]['SNNI'].unique())))

    # Determine the common y-axis limits based on the data
    all_values = pd.concat([data['Value'] for data in benchmark_data.values()])
    y_min, y_max = all_values.min(), all_values.max()

    # Add padding to y-axis limits
    y_padding = 0.1 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    for ax, (benchmark, data) in zip(axs.flat, benchmark_data.items()):
        if data.empty:
            print(f"No data available for {benchmark}, skipping plot.")
            continue

        for i, (snni, snni_data) in enumerate(data.groupby('SNNI')):
            ax.scatter(snni, snni_data['Value'], color=snni_colors[i], s=100, alpha=0.8, edgecolors="w", linewidth=2, label=snni)

        ax.set_title(f'{benchmark} - CPU Usage')
        ax.set_xlabel('SNNI Approaches')
        ax.set_ylabel('Average CPU Usage (%)')
        ax.set_xticks(np.arange(len(data['SNNI'].unique())))
        ax.set_xticklabels(data['SNNI'].unique(), rotation=45)
        ax.set_ylim(y_min, y_max)  # Set the same y-axis limits for all plots with padding

    # Hide any unused subplots
    for ax in axs.flat[len(benchmark_data):]:
        ax.axis('off')

    # Add a single legend for all subplots, positioned below the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="SNNI Approaches", loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize='small')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust rect to make room for the legend

    # Save plot as JPEG
    plt.savefig(output_filename, format='jpeg')
    plt.close()

# Define paths to CSV files for each benchmark in different SNNI approaches

benchmark_paths = {
    'AlexNet': {
        'Cheetah': 'cheetah_alexnet_client_metrics.csv',
        'SCI_HE': 'SCI_HE_alexnet_client_metrics.csv',
        'Porthos': 'porthos_AlexNet_party0_metrics.csv',
        'SCI': 'SCI_AlexNet_client_metrics.csv'
    },
    'DenseNet121': {
        'Cheetah': 'cheetah_densenet121_client_metrics.csv',
        'SCI_HE': 'SCI_HE_densenet121_client_metrics.csv',
        'Porthos': 'porthos_DenseNet_party0_metrics.csv',
        'SCI': 'SCI_DenseNet_client_metrics.csv'
    },
    'LeNet': {
        'Cheetah': 'cheetah_lenet-large_client_metrics.csv',
        'SCI_HE': 'SCI_HE_lenet-large_client_metrics.csv',
        'Porthos': 'porthos_Lenet-large_party0_metrics.csv',
        'SCI': 'SCI_Lenet-large_client_metrics.csv'
    },
    'ResNet50': {
        'Cheetah': 'cheetah_resnet50_client_metrics.csv',
        'SCI_HE': 'SCI_HE_resnet50_client_metrics.csv',
        'Porthos': 'porthos_ResNet_party0_metrics.csv',
        'SCI': 'SCI_ResNet_client_metrics.csv'
    },
    'ShuffleNetV2': {
        'Cheetah': 'cheetah_shufflenetv2_client_metrics.csv',
        'SCI_HE': 'SCI_HE_shufflenetv2_client_metrics.csv',
        'Porthos': 'porthos_ShuffleNetV2_party0_metrics.csv',
        'SCI': 'SCI_ShuffleNetV2_client_metrics.csv'
    },
    'SqueezeNet-ImgNet': {
        'Cheetah': 'cheetah_sqnet_client_metrics.csv',
        'SCI_HE': 'SCI_HE_sqnet_client_metrics.csv',
        'Porthos': 'porthos_SqueezeNetImgNet_party0_metrics.csv',
        'SCI': 'SCI_SqueezeNetImgNet_client_metrics.csv'
    },
    'SqueezeNet-CIFAR10': {
        'Cheetah': 'cheetah_SqueezeNetCIFAR10_client_metrics.csv',
        'SCI_HE': 'SCI_HE_SqueezeNetCIFAR10_client_metrics.csv',
        'Porthos': 'porthos_SqueezeNetCIFAR10_party0_metrics.csv',
        'SCI': 'SCI_SqueezeNetCIFAR10_client_metrics.csv'
    }
}

# Process CPU data for each benchmark across different SNNI approaches
benchmark_cpu_data = {benchmark: process_cpu_data_by_benchmark(paths, benchmark) for benchmark, paths in benchmark_paths.items()}

# Plot CPU usage for each benchmark across different SNNI approaches in one combined plot
plot_cpu_usage_by_benchmark(benchmark_cpu_data, 'Average CPU Usage of Client Across Benchmarks in Different SNNI Approaches', 'combined_benchmark_cpu_usage.jpeg')
