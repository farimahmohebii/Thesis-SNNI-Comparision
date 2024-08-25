import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_cpu_data(paths):
    all_data = []

    for model, path in paths.items():
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"The file {path} is empty.")
            print(f"Columns in {path}: {df.columns.tolist()}")

            # Extract only the average CPU usage data
            cpu_data = df[df['Metric'] == 'Average CPU usage (%)']
            cpu_data['Model'] = model
            all_data.append(cpu_data)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    if all_data:
        aggregated_data = pd.concat(all_data, ignore_index=True)
        return aggregated_data
    else:
        print("No data was aggregated.")
        return pd.DataFrame()

def plot_cpu_usage_combined(cheetah_df, sci_he_df, porthos_df, sci_df, title, output_filename):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    snni_data = {
        'Cheetah': cheetah_df,
        'SCI_HE': sci_he_df,
        'Porthos': porthos_df,
        'SCI': sci_df
    }
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(snni_data['Cheetah']['Model'].unique())))

    for ax, (snni, df) in zip(axs.flat, snni_data.items()):
        if df.empty:
            print(f"{snni} DataFrame is empty, skipping plot.")
            continue
        
        for i, model in enumerate(df['Model'].unique()):
            model_data = df[df['Model'] == model]
            ax.scatter(model_data['Model'], model_data['Value'], color=colors[i], s=100, alpha=0.8, edgecolors="w", linewidth=2, label=model)

        ax.set_title(f'{snni} - Average CPU Usage')
        ax.set_xlabel('Benchmarks')
        ax.set_ylabel('Average CPU Usage (%)')
        ax.set_xticks(np.arange(len(df['Model'].unique())))
        ax.set_xticklabels(df['Model'].unique(), rotation=45)

    # Add a single legend for all subplots, positioned below the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Benchmarks", loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, fontsize='small')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust rect to make room for the legend

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

porthos_party0_path = {
    'AlexNet': 'porthos_AlexNet_party0_metrics.csv',
    'DenseNet121': 'porthos_DenseNet_party0_metrics.csv',
    'LeNet': 'porthos_Lenet-large_party0_metrics.csv',
    'ResNet50': 'porthos_ResNet_party0_metrics.csv',
    'ShuffleNetV2': 'porthos_ShuffleNetV2_party0_metrics.csv',
    'SqueezeNet-CIFAR10': 'porthos_SqueezeNetCIFAR10_party0_metrics.csv',
    'SqueezeNet-ImgNet': 'porthos_SqueezeNetImgNet_party0_metrics.csv'
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

# Process CPU data for each SNNI approach
cheetah_cpu_data = process_cpu_data(cheetah_client_path)
sci_he_cpu_data = process_cpu_data(sci_he_client_path)
porthos_cpu_data = process_cpu_data(porthos_party0_path)
sci_cpu_data = process_cpu_data(sci_client_path)

# Plot all SNNI approaches in one plot with 4 subplots
plot_cpu_usage_combined(cheetah_cpu_data, sci_he_cpu_data, porthos_cpu_data, sci_cpu_data, 'Average CPU Usage Across SNNI Approaches', 'combined_cpu_usage.jpeg')
