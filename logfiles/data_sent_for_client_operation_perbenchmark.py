import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the paths to client CSV files excluding Porthos
csv_files = {
    'AlexNet': {
        'Cheetah': 'cheetah_alexnet_client_metrics.csv',
        'SCI_HE': 'SCI_HE_alexnet_client_metrics.csv',
        'SCI': 'SCI_AlexNet_client_metrics.csv',
    },
    'DenseNet121': {
        'Cheetah': 'cheetah_densenet121_client_metrics.csv',
        'SCI_HE': 'SCI_HE_densenet121_client_metrics.csv',
        'SCI': 'SCI_DenseNet_client_metrics.csv',
    },
    'LeNet': {
        'Cheetah': 'cheetah_lenet-large_client_metrics.csv',
        'SCI_HE': 'SCI_HE_lenet-large_client_metrics.csv',
        'SCI': 'SCI_Lenet-large_client_metrics.csv',
    },
    'ResNet50': {
        'Cheetah': 'cheetah_resnet50_client_metrics.csv',
        'SCI_HE': 'SCI_HE_resnet50_client_metrics.csv',
        'SCI': 'SCI_ResNet_client_metrics.csv',
    },
    'ShuffleNetV2': {
        'Cheetah': 'cheetah_shufflenetv2_client_metrics.csv',
        'SCI_HE': 'SCI_HE_shufflenetv2_client_metrics.csv',
        'SCI': 'SCI_ShuffleNetV2_client_metrics.csv',
    },
    'SqueezeNet_ImgNet': {
        'Cheetah': 'cheetah_sqnet_client_metrics.csv',
        'SCI_HE': 'SCI_HE_sqnet_client_metrics.csv',
        'SCI': 'SCI_SqueezeNetImgNet_client_metrics.csv',
    },
    'SqueezeNet_CIFAR10': {
        'Cheetah': 'cheetah_SqueezeNetCIFAR10_client_metrics.csv',
        'SCI_HE': 'SCI_HE_SqueezeNetCIFAR10_client_metrics.csv',
        'SCI': 'SCI_SqueezeNetCIFAR10_client_metrics.csv',
    }
}

# Define the operations of interest
operations = [
    'Conv data sent (MiB)', 'MatMul data sent (MiB)', 'BatchNorm data sent (MiB)', 
    'Truncation data sent (MiB)', 'Relu data sent (MiB)', 'Maxpool data sent (MiB)',
    'Avgpool data sent (MiB)', 'ArgMax data sent (MiB)'
]

# Loop over each benchmark and plot the data sent for different operations
for benchmark, paths in csv_files.items():
    # Initialize a DataFrame to hold the data for the current benchmark
    benchmark_df = pd.DataFrame()

    # Gather data from all SNNI approaches
    for snni, file_path in paths.items():
        df = pd.read_csv(file_path)
        df = df[df['Metric'].isin(operations)]
        df['SNNI'] = snni  # Add SNNI approach as a column
        benchmark_df = pd.concat([benchmark_df, df])

    # Pivot the data for easy plotting
    pivot_df = benchmark_df.pivot(index='Metric', columns='SNNI', values='Value')

    # Set up the figure for each benchmark
    plt.figure(figsize=(14, 10))  # Further increase figure size to give more space
    ax = pivot_df.plot(kind='bar', colormap='viridis', logy=True, figsize=(14, 8))  # Log scale for y-axis

    plt.title(f'{benchmark} - Data Sent for Client in Different Operations')
    plt.ylabel('Data Sent (MiB)')
    plt.xticks(np.arange(len(pivot_df.index)), pivot_df.index, rotation=45, ha='right')  # Rotate labels and adjust alignment
    
    # Move the legend outside the plot
    plt.legend(title="SNNI Approaches", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add more spacing around the plot
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.25)
    
    # Save each figure as a JPEG file
    plt.savefig(f'{benchmark}_data_sent_client_updated.jpeg', format='jpeg')
    plt.close()

