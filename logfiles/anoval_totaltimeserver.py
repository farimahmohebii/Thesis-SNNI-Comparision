import pandas as pd
from scipy.stats import f_oneway

# Define paths to CSV files for each SNNI approach
csv_paths = {
    'Cheetah': [
        'cheetah_alexnet_server_metrics.csv',
        'cheetah_densenet121_server_metrics.csv',
        'cheetah_lenet-large_server_metrics.csv',
        'cheetah_resnet50_server_metrics.csv',
        'cheetah_shufflenetv2_server_metrics.csv',
        'cheetah_sqnet_server_metrics.csv',
        'cheetah_SqueezeNetCIFAR10_server_metrics.csv'
    ],
    'SCI_HE': [
        'SCI_HE_alexnet_server_metrics.csv',
        'SCI_HE_densenet121_server_metrics.csv',
        'SCI_HE_lenet-large_server_metrics.csv',
        'SCI_HE_resnet50_server_metrics.csv',
        'SCI_HE_shufflenetv2_server_metrics.csv',
        'SCI_HE_sqnet_server_metrics.csv',
        'SCI_HE_SqueezeNetCIFAR10_server_metrics.csv'
    ],
    'Porthos Party1': [
        'porthos_AlexNet_party1_metrics.csv',
        'porthos_DenseNet_party1_metrics.csv',
        'porthos_Lenet-large_party1_metrics.csv',
        'porthos_ResNet_party1_metrics.csv',
        'porthos_ShuffleNetV2_party1_metrics.csv',
        'porthos_SqueezeNetImgNet_party1_metrics.csv',
        'porthos_SqueezeNetCIFAR10_party1_metrics.csv'
    ],
    'SCI': [
        'SCI_AlexNet_server_metrics.csv',
        'SCI_DenseNet_server_metrics.csv',
        'SCI_Lenet-large_server_metrics.csv',
        'SCI_ResNet_server_metrics.csv',
        'SCI_ShuffleNetV2_server_metrics.csv',
        'SCI_SqueezeNetImgNet_server_metrics.csv',
        'SCI_SqueezeNetCIFAR10_server_metrics.csv'
    ]
}

# Function to extract 'Total time taken (ms)' for each benchmark and SNNI approach
def extract_total_time_taken(csv_paths):
    total_times = {benchmark: [] for benchmark in range(7)}
    
    for snni, paths in csv_paths.items():
        for i, path in enumerate(paths):
            try:
                df = pd.read_csv(path)
                if df.empty:
                    raise ValueError(f"The file {path} is empty.")
                total_time = df[df['Metric'] == 'Total time taken (ms)']['Value'].astype(float).values[0]
                total_times[i].append(total_time)
            except Exception as e:
                print(f"Failed to process {path}: {e}")
    
    return total_times

# Extract data
total_times = extract_total_time_taken(csv_paths)

# Perform ANOVA
for i, (benchmark, times) in enumerate(total_times.items()):
    if len(times) == 4:  # Ensure all 4 SNNI approaches have data
        f_stat, p_val = f_oneway(times[0], times[1], times[2], times[3])
        print(f"ANOVA results for Benchmark {i+1}: F-Statistic = {f_stat}, P-Value = {p_val}")
    else:
        print(f"Insufficient data for Benchmark {i+1}")
