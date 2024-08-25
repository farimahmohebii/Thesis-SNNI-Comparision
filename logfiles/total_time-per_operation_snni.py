import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the paths to client CSV files excluding Porthos
csv_files = {
    'Cheetah': {
        'AlexNet': 'cheetah_alexnet_client_metrics.csv',
        'DenseNet121': 'cheetah_densenet121_client_metrics.csv',
        'LeNet': 'cheetah_lenet-large_client_metrics.csv',
        'ResNet50': 'cheetah_resnet50_client_metrics.csv',
        'ShuffleNetV2': 'cheetah_shufflenetv2_client_metrics.csv',
        'SqueezeNet_ImgNet': 'cheetah_sqnet_client_metrics.csv',
        'SqueezeNet_CIFAR10': 'cheetah_SqueezeNetCIFAR10_client_metrics.csv',
    },
    'SCI_HE': {
        'AlexNet': 'SCI_HE_alexnet_client_metrics.csv',
        'DenseNet121': 'SCI_HE_densenet121_client_metrics.csv',
        'LeNet': 'SCI_HE_lenet-large_client_metrics.csv',
        'ResNet50': 'SCI_HE_resnet50_client_metrics.csv',
        'ShuffleNetV2': 'SCI_HE_shufflenetv2_client_metrics.csv',
        'SqueezeNet_ImgNet': 'SCI_HE_sqnet_client_metrics.csv',
        'SqueezeNet_CIFAR10': 'SCI_HE_SqueezeNetCIFAR10_client_metrics.csv',
    },
    'SCI': {
        'AlexNet': 'SCI_AlexNet_client_metrics.csv',
        'DenseNet121': 'SCI_DenseNet_client_metrics.csv',
        'LeNet': 'SCI_Lenet-large_client_metrics.csv',
        'ResNet50': 'SCI_ResNet_client_metrics.csv',
        'ShuffleNetV2': 'SCI_ShuffleNetV2_client_metrics.csv',
        'SqueezeNet_ImgNet': 'SCI_SqueezeNetImgNet_client_metrics.csv',
        'SqueezeNet_CIFAR10': 'SCI_SqueezeNetCIFAR10_client_metrics.csv',
    }
}

# Define the operations of interest in seconds (s)
operations_time = [
    'Total time in Conv (s)', 'Total time in MatMul (s)', 'Total time in BatchNorm (s)', 
    'Total time in Truncation (s)', 'Total time in Relu (s)', 'Total time in MaxPool (s)',
    'Total time in AvgPool (s)', 'Total time in ArgMax (s)'
]

# Loop over each SNNI approach and plot the data for different benchmarks
for snni, benchmarks in csv_files.items():
    # Initialize a DataFrame to hold the data for the current SNNI approach
    snni_df = pd.DataFrame()

    # Gather data from all benchmarks
    for benchmark, file_path in benchmarks.items():
        df = pd.read_csv(file_path)
        df = df[df['Metric'].isin(operations_time)]
        if df.empty:
            print(f"No numeric data found for {benchmark} in {snni}. Skipping this benchmark.")
            continue
        df['Benchmark'] = benchmark  # Add benchmark as a column
        snni_df = pd.concat([snni_df, df])

    if snni_df.empty:
        print(f"No data to plot for {snni}.")
        continue

    # Pivot the data for easy plotting
    pivot_df = snni_df.pivot(index='Metric', columns='Benchmark', values='Value')

    # Set up the figure for each SNNI approach
    plt.figure(figsize=(14, 10))  # Increase figure size
    ax = pivot_df.plot(kind='bar', colormap='viridis', logy=True, figsize=(14, 8))  # Log scale for y-axis

    plt.title(f'{snni} - Total Time Taken for Client in Different Operations')
    plt.ylabel('Total Time Taken (s)')
    plt.xticks(np.arange(len(pivot_df.index)), pivot_df.index, rotation=45, ha='right')  # Rotate labels and adjust alignment
    
    # Move the legend outside the plot and adjust the position
    plt.legend(title="Benchmarks", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')

    # Adjust layout to ensure everything fits and the legend is visible
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.25)
    
    # Save each figure as a JPEG file
    plt.savefig(f'{snni}_total_time_per_operation_client_updated.jpeg', format='jpeg')
    plt.close()
