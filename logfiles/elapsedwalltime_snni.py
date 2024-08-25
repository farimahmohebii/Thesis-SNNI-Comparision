import pandas as pd
import matplotlib.pyplot as plt

# Define the paths to client CSV files including Porthos Party0
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
    },
    'PorthosParty0': {
        'AlexNet': 'porthos_AlexNet_party0_metrics.csv',
        'DenseNet121': 'porthos_DenseNet_party0_metrics.csv',
        'LeNet': 'porthos_Lenet-large_party0_metrics.csv',
        'ResNet50': 'porthos_ResNet_party0_metrics.csv',
        'ShuffleNetV2': 'porthos_ShuffleNetV2_party0_metrics.csv',
        'SqueezeNet_ImgNet': 'porthos_SqueezeNetImgNet_party0_metrics.csv',
        'SqueezeNet_CIFAR10': 'porthos_SqueezeNetCIFAR10_party0_metrics.csv',
    }
}

# Define the metric of interest
metric_of_interest = 'Elapsed wall time (s)'

# Loop over each SNNI approach and plot the data for different benchmarks
for snni, benchmarks in csv_files.items():
    # Initialize a DataFrame to hold the data for the current SNNI approach
    snni_df = pd.DataFrame()

    # Gather data from all benchmarks
    for benchmark, file_path in benchmarks.items():
        df = pd.read_csv(file_path)
        df = df[df['Metric'] == metric_of_interest]
        if df.empty:
            print(f"No data for {metric_of_interest} in {benchmark} for {snni}. Skipping this benchmark.")
            continue
        df['Benchmark'] = benchmark  # Add benchmark as a column
        snni_df = pd.concat([snni_df, df])

    if snni_df.empty:
        print(f"No data to plot for {snni}.")
        continue

    # Pivot the data for easy plotting
    pivot_df = snni_df.pivot(index='Benchmark', columns='Metric', values='Value')

    # Set up the figure for each SNNI approach
    plt.figure(figsize=(10, 6))
    ax = pivot_df.plot(kind='bar', colormap='viridis', figsize=(10, 6))

    plt.title(f'{snni} - Elapsed Wall Time for Client in Different Benchmarks')
    plt.ylabel('Elapsed Wall Time (s)')
    plt.xticks(rotation=45, ha='right')

    # Move the legend outside the plot and adjust the position
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust layout to ensure everything fits and the legend is visible
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save each figure as a JPEG file
    plt.savefig(f'{snni}_elapsed_wall_time_per_benchmark.jpeg', format='jpeg')
    plt.close()
