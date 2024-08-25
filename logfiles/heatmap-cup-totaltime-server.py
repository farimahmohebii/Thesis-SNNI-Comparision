import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the paths to all relevant server CSV files
csv_files = {
    'Cheetah_AlexNet': 'cheetah_alexnet_server_metrics.csv',
    'SCI_HE_AlexNet': 'SCI_HE_alexnet_server_metrics.csv',
    'Porthos_AlexNet': 'porthos_AlexNet_party1_metrics.csv',
    'SCI_AlexNet': 'SCI_AlexNet_server_metrics.csv',
    'Cheetah_DenseNet121': 'cheetah_densenet121_server_metrics.csv',
    'SCI_HE_DenseNet121': 'SCI_HE_densenet121_server_metrics.csv',
    'Porthos_DenseNet121': 'porthos_DenseNet_party1_metrics.csv',
    'SCI_DenseNet121': 'SCI_DenseNet_server_metrics.csv',
    'Cheetah_LeNet': 'cheetah_lenet-large_server_metrics.csv',
    'SCI_HE_LeNet': 'SCI_HE_lenet-large_server_metrics.csv',
    'Porthos_LeNet': 'porthos_Lenet-large_party1_metrics.csv',
    'SCI_LeNet': 'SCI_Lenet-large_server_metrics.csv',
    'Cheetah_ResNet50': 'cheetah_resnet50_server_metrics.csv',
    'SCI_HE_ResNet50': 'SCI_HE_resnet50_server_metrics.csv',
    'Porthos_ResNet50': 'porthos_ResNet_party1_metrics.csv',
    'SCI_ResNet50': 'SCI_ResNet_server_metrics.csv',
    'Cheetah_ShuffleNetV2': 'cheetah_shufflenetv2_server_metrics.csv',
    'SCI_HE_ShuffleNetV2': 'SCI_HE_shufflenetv2_server_metrics.csv',
    'Porthos_ShuffleNetV2': 'porthos_ShuffleNetV2_party1_metrics.csv',
    'SCI_ShuffleNetV2': 'SCI_ShuffleNetV2_server_metrics.csv',
    'Cheetah_SqueezeNet_ImgNet': 'cheetah_sqnet_server_metrics.csv',
    'SCI_HE_SqueezeNet_ImgNet': 'SCI_HE_sqnet_server_metrics.csv',
    'Porthos_SqueezeNet_ImgNet': 'porthos_SqueezeNetImgNet_party1_metrics.csv',
    'SCI_SqueezeNet_ImgNet': 'SCI_SqueezeNetImgNet_server_metrics.csv',
    'Cheetah_SqueezeNet_CIFAR10': 'cheetah_SqueezeNetCIFAR10_server_metrics.csv',
    'SCI_HE_SqueezeNet_CIFAR10': 'SCI_HE_SqueezeNetCIFAR10_server_metrics.csv',
    'Porthos_SqueezeNet_CIFAR10': 'porthos_SqueezeNetCIFAR10_party1_metrics.csv',
    'SCI_SqueezeNet_CIFAR10': 'SCI_SqueezeNetCIFAR10_server_metrics.csv',
}

# Initialize an empty DataFrame to hold all metrics
all_metrics_df = pd.DataFrame()

# Loop over each CSV file and extract relevant metrics
for model_name, file_path in csv_files.items():
    df = pd.read_csv(file_path)
    
    # Extract the relevant metrics
    metrics_of_interest = df[df['Metric'].isin([
        'Total time taken (ms)', 
        'Average CPU usage (%)', 
        'Estimated energy used (J)'
    ])]
    
    # Pivot the DataFrame to have metrics as columns and model as index
    metrics_pivot = metrics_of_interest.pivot_table(index=None, columns='Metric', values='Value').reset_index(drop=True)
    
    # Add columns for the model and SNNI approach
    snni, model = model_name.split('_', 1)
    metrics_pivot['Model'] = model
    metrics_pivot['SNNI'] = snni
    
    # Append to the overall DataFrame
    all_metrics_df = pd.concat([all_metrics_df, metrics_pivot], ignore_index=True)

# Calculate the correlation matrix for each SNNI approach
for snni in all_metrics_df['SNNI'].unique():
    snni_df = all_metrics_df[all_metrics_df['SNNI'] == snni]
    correlation_matrix = snni_df.drop(columns=['Model', 'SNNI']).corr()
    
    # Plot the heatmap for each SNNI approach
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Heatmap of Metrics Across Benchmarks ({snni} - Server)')
    plt.show()
    
    # Save the heatmap as a JPEG file
    plt.savefig(f'{snni}_server_metrics_correlation_heatmap.jpeg', format='jpeg')
