import pandas as pd
import matplotlib.pyplot as plt

# Define the specific data sent metrics you're interested in
expected_metrics_data_sent = [
    'Conv data sent (MiB)', 'MatMul data sent (MiB)', 'BatchNorm data sent (MiB)',
    'Truncation data sent (MiB)', 'Relu data sent (MiB)', 'Maxpool data sent (MiB)',
    'Avgpool data sent (MiB)', 'ArgMax data sent (MiB)'
]

def process_and_aggregate(paths):
    all_data = []

    for model, path in paths.items():
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"The file {path} is empty.")
            print(f"Columns in {path}: {df.columns.tolist()}")

            # Filter for data sent metrics only
            df_filtered = df[df['Metric'].isin(expected_metrics_data_sent)]
            if df_filtered.empty:
                raise ValueError(f"None of the expected metrics are found in {path}.")
            
            df_filtered['Model'] = model
            all_data.append(df_filtered)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    if all_data:
        aggregated_data = pd.concat(all_data, ignore_index=True)
        return aggregated_data
    else:
        print("No data was aggregated.")
        return pd.DataFrame()

def plot_aggregated_data(df, title, output_filename):
    if df.empty:
        print("DataFrame is empty, no data to plot.")
        return

    # Pivot the DataFrame so that each metric is a category on the x-axis
    df_pivot = df.pivot(index='Metric', columns='Model', values='Value')

    ax = df_pivot.plot(kind='bar', figsize=(14, 8), colormap='viridis')
    ax.set_yscale('log')  # Apply logarithmic scale to the y-axis
    ax.set_title(title)
    ax.set_ylabel('Data Sent (MiB)')
    ax.set_xlabel('Operations')
    plt.xticks(rotation=45)
    plt.legend(title='Models')
    plt.tight_layout()

    # Save plot as JPEG
    plt.savefig(output_filename, format='jpeg')
    plt.close()

# Define paths to CSV files for each SNNI approach

cheetah_client_path = {
    'AlexNet': 'cheetah_alexnet_client_metrics.csv',
    'DenseNet121': 'cheetah_densenet121_client_metrics.csv',
    'LeNet': 'cheetah_lenet-large_client_metrics.csv',
    'ResNet50': 'cheetah_resnet50_client_metrics.csv',
    'ShuffleNetV2': 'cheetah_shufflenetv2_client_metrics.csv',
    'SqueezeNet-ImgNet': 'cheetah_sqnet_client_metrics.csv',
    'SqueezeNet-CIFAR10': 'cheetah_SqueezeNetCIFAR10_client_metrics.csv'
}

sci_he_client_path = {
    'AlexNet': 'SCI_HE_alexnet_client_metrics.csv',
    'DenseNet121': 'SCI_HE_densenet121_client_metrics.csv',
    'LeNet': 'SCI_HE_lenet-large_client_metrics.csv',
    'ResNet50': 'SCI_HE_resnet50_client_metrics.csv',
    'ShuffleNetV2': 'SCI_HE_shufflenetv2_client_metrics.csv',
    'SqueezeNet-CIFAR10': 'SCI_HE_SqueezeNetCIFAR10_client_metrics.csv',
    'SqueezeNet-ImgNet': 'SCI_HE_sqnet_client_metrics.csv'
}

sci_client_path = {
    'AlexNet': 'SCI_AlexNet_client_metrics.csv',
    'DenseNet121': 'SCI_DenseNet_client_metrics.csv',
    'LeNet': 'SCI_Lenet-large_client_metrics.csv',
    'ResNet50': 'SCI_ResNet_client_metrics.csv',
    'ShuffleNetV2': 'SCI_ShuffleNetV2_client_metrics.csv',
    'SqueezeNet-CIFAR10': 'SCI_SqueezeNetCIFAR10_client_metrics.csv',
    'SqueezeNet-ImgNet': 'SCI_SqueezeNetImgNet_client_metrics.csv'
}


# Process and plot data for Cheetah client
cheetah_client_data = process_and_aggregate(cheetah_client_path)
plot_aggregated_data(cheetah_client_data, 'Cheetah Client - Data Sent by Operation Across Models', 'cheetah_data_sent_plot.jpeg')

# Process and plot data for SCI_HE client
sci_he_client_data = process_and_aggregate(sci_he_client_path)
plot_aggregated_data(sci_he_client_data, 'SCI_HE Client - Data Sent by Operation Across Models', 'sci_he_data_sent_plot.jpeg')


# Process and plot data for SCI client
sci_client_data = process_and_aggregate(sci_client_path)
plot_aggregated_data(sci_client_data, 'SCI Client - Data Sent by Operation Across Models', 'sci_data_sent_plot.jpeg')

