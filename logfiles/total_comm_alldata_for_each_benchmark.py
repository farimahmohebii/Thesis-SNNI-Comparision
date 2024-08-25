import pandas as pd
import matplotlib.pyplot as plt

def process_data(paths, is_porthos=False):
    data_frames = []
    for model, path in paths.items():
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"The file {path} is empty.")
            # Extract 'Total comm (sent+received)' and apply conversion if needed
            df_filtered = df[df['Metric'] == 'Total comm (sent+received) (MiB)'].copy()
            if is_porthos:  # Specific normalization for Porthos
                df_time = df[df['Metric'] == 'Total time taken (ms)'].copy()
                if not df_time.empty:
                    df_filtered['Value'] = (df_filtered['Value'] * 1e6) / (df_time['Value'].iloc[0] * 1000 * 1000)
            df_filtered['SNNI'] = model
            data_frames.append(df_filtered[['SNNI', 'Value']])
        except Exception as e:
            print(f"Failed to process {path}: {e}")
    return pd.concat(data_frames, ignore_index=True)

def plot_data(data, benchmark, output_filename):
    if data.empty:
        print("No data to plot.")
        return

    ax = data.plot(kind='bar', x='SNNI', y='Value', legend=False, color=['blue', 'green', 'red', 'purple'])
    ax.set_title(f'Total Comm (Sent+Received) for {benchmark}')
    ax.set_ylabel('Total Comm (MiB)')
    ax.set_xlabel('SNNI Approaches')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot as JPEG
    plt.savefig(output_filename, format='jpeg')
    plt.close()

# Paths for each SNNI approach and benchmark
server_paths = {
    'Cheetah': {
        'AlexNet': 'cheetah_alexnet_server_metrics.csv',
        'DenseNet121': 'cheetah_densenet121_server_metrics.csv',
        'LeNet': 'cheetah_lenet-large_server_metrics.csv',
        'ResNet50': 'cheetah_resnet50_server_metrics.csv',
        'ShuffleNetV2': 'cheetah_shufflenetv2_server_metrics.csv',
        'SqueezeNet-ImgNet': 'cheetah_sqnet_server_metrics.csv',
        'SqueezeNet-CIFAR10': 'cheetah_SqueezeNetCIFAR10_server_metrics.csv'
    },
    'SCI_HE': {
        'AlexNet': 'SCI_HE_alexnet_server_metrics.csv',
        'DenseNet121': 'SCI_HE_densenet121_server_metrics.csv',
        'LeNet': 'SCI_HE_lenet-large_server_metrics.csv',
        'ResNet50': 'SCI_HE_resnet50_server_metrics.csv',
        'ShuffleNetV2': 'SCI_HE_shufflenetv2_server_metrics.csv',
        'SqueezeNet-CIFAR10': 'SCI_HE_SqueezeNetCIFAR10_server_metrics.csv',
        'SqueezeNet-ImgNet': 'SCI_HE_sqnet_server_metrics.csv'
    },
    'SCI': {
        'AlexNet': 'SCI_AlexNet_server_metrics.csv',
        'DenseNet121': 'SCI_DenseNet_server_metrics.csv',
        'LeNet': 'SCI_Lenet-large_server_metrics.csv',
        'ResNet50': 'SCI_ResNet_server_metrics.csv',
        'ShuffleNetV2': 'SCI_ShuffleNetV2_server_metrics.csv',
        'SqueezeNet-CIFAR10': 'SCI_SqueezeNetCIFAR10_server_metrics.csv',
        'SqueezeNet-ImgNet': 'SCI_SqueezeNetImgNet_server_metrics.csv'
    },
    'PorthosParty1': {
        'AlexNet': 'porthos_AlexNet_party1_metrics.csv',
        'DenseNet121': 'porthos_DenseNet_party1_metrics.csv',
        'LeNet': 'porthos_Lenet-large_party1_metrics.csv',
        'ResNet50': 'porthos_ResNet_party1_metrics.csv',
        'ShuffleNetV2': 'porthos_ShuffleNetV2_party1_metrics.csv',
        'SqueezeNet-CIFAR10': 'porthos_SqueezeNetCIFAR10_party1_metrics.csv',
        'SqueezeNet-ImgNet': 'porthos_SqueezeNetImgNet_party1_metrics.csv'
    }
}

# Process and plot data for each benchmark and SNNI approach
for benchmark, paths in server_paths.items():
    is_porthos = 'PorthosParty1' in paths.keys()
    data = process_data(paths, is_porthos)
    plot_data(data, benchmark, f'{benchmark}_total_comm_server.jpeg')
