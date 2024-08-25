import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the paths to server CSV files and Porthos Party1 files with "pm" added before ".csv"
csv_files = {
    'Cheetah': {
        'AlexNet': 'cheetah_alexnet_server_metricspm.csv',
        'DenseNet121': 'cheetah_densenet121_server_metricspm.csv',
        'LeNet': 'cheetah_lenet-large_server_metricspm.csv',
        'ResNet50': 'cheetah_resnet50_server_metricspm.csv',
        'ShuffleNetV2': 'cheetah_shufflenetv2_server_metricspm.csv',
        'SqueezeNet-ImgNet': 'cheetah_sqnet_server_metricspm.csv',
        'SqueezeNet-CIFAR10': 'cheetah_SqueezeNetCIFAR10_server_metricspm.csv'
    },
    'SCI_HE': {
        'AlexNet': 'SCI_HE_alexnet_server_metricspm.csv',
        'DenseNet121': 'SCI_HE_densenet121_server_metricspm.csv',
        'LeNet': 'SCI_HE_lenet-large_server_metricspm.csv',
        'ResNet50': 'SCI_HE_resnet50_server_metricspm.csv',
        'ShuffleNetV2': 'SCI_HE_shufflenetv2_server_metricspm.csv',
        'SqueezeNet-CIFAR10': 'SCI_HE_SqueezeNetCIFAR10_server_metricspm.csv',
        'SqueezeNet-ImgNet': 'SCI_HE_sqnet_server_metricspm.csv'
    },
    'SCI': {
        'AlexNet': 'SCI_AlexNet_server_metricspm.csv',
        'DenseNet121': 'SCI_DenseNet_server_metricspm.csv',
        'LeNet': 'SCI_Lenet-large_server_metricspm.csv',
        'ResNet50': 'SCI_ResNet_server_metricspm.csv',
        'ShuffleNetV2': 'SCI_ShuffleNetV2_server_metricspm.csv',
        'SqueezeNet-CIFAR10': 'SCI_SqueezeNetCIFAR10_server_metricspm.csv',
        'SqueezeNet-ImgNet': 'SCI_SqueezeNetImgNet_server_metricspm.csv'
    },
    'Porthos': {
        'AlexNet': 'porthos_AlexNet_party1_metricspm.csv',
        'DenseNet121': 'porthos_DenseNet_party1_metricspm.csv',
        'LeNet': 'porthos_Lenet-large_party1_metricspm.csv',
        'ResNet50': 'porthos_ResNet_party1_metricspm.csv',
        'ShuffleNetV2': 'porthos_ShuffleNetV2_party1_metricspm.csv',
        'SqueezeNet-CIFAR10': 'porthos_SqueezeNetCIFAR10_party1_metricspm.csv',
        'SqueezeNet-ImgNet': 'porthos_SqueezeNetImgNet_party1_metricspm.csv'
    }
}

# Define the metric of interest
metric_of_interest = 'Estimated energy used (J)'

# Prepare DataFrame to collect all data
all_data = []

# Loop over each SNNI approach and benchmark to collect data
for snni, benchmarks in csv_files.items():
    for benchmark, file_path in benchmarks.items():
        try:
            df = pd.read_csv(file_path)
            df = df[df['Metric'] == metric_of_interest]
            if df.empty:
                print(f"No data for {metric_of_interest} in {benchmark} for {snni}. Skipping this benchmark.")
                continue

            df['Value'] = df['Value'] / 8  # Divide the energy used by 8
            df['Benchmark'] = benchmark  # Add benchmark as a column
            df['SNNI'] = snni  # Add SNNI approach as a column
            all_data.append(df[['Benchmark', 'SNNI', 'Value']])
        
        except FileNotFoundError as e:
            print(f"File not found: {file_path}. Skipping...")
            continue

# Combine all data into a single DataFrame
if all_data:
    combined_df = pd.concat(all_data)
else:
    print("No data to plot.")
    exit()

# Pivot the data for grouped bar plotting
pivot_df = combined_df.pivot(index='Benchmark', columns='SNNI', values='Value')

# Plotting
plt.figure(figsize=(14, 8))
ax = pivot_df.plot(kind='bar', colormap='viridis', figsize=(14, 8), logy=True)  # Logarithmic scale for y-axis

plt.title('Estimated Energy Used (J) ')
plt.ylabel('Estimated Energy Used (J)')
plt.xticks(rotation=45, ha='right')

# Move the legend outside the plot and adjust the position
plt.legend(title="SNNI Approaches", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Adjust layout to ensure everything fits and the legend is visible
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save the figure as a JPEG file
plt.savefig('estimated_energy_used_divided_by_8_across_snni_logscale.jpeg', format='jpeg')
plt.close()
