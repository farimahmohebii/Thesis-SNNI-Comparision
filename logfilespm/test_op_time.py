import pandas as pd
from scipy.stats import wilcoxon

# Define paths to VM and PM client files (including Porthos Party0)
vm_files = {
    'AlexNet': '/home/mohebifarimah/logfiles/SCI_AlexNet_client_metrics.csv',
    'DenseNet121': '/home/mohebifarimah/logfiles/SCI_DenseNet_client_metrics.csv',
    'LeNet': '/home/mohebifarimah/logfiles/SCI_Lenet-large_client_metrics.csv',
    'ResNet50': '/home/mohebifarimah/logfiles/SCI_ResNet_client_metrics.csv',
    'ShuffleNetV2': '/home/mohebifarimah/logfiles/SCI_ShuffleNetV2_client_metrics.csv',
    'SqueezeNet_ImgNet': '/home/mohebifarimah/logfiles/SCI_SqueezeNetImgNet_client_metrics.csv',
    'SqueezeNet_CIFAR10': '/home/mohebifarimah/logfiles/SCI_SqueezeNetCIFAR10_client_metrics.csv',
    'Porthos_AlexNet': '/home/mohebifarimah/logfiles/porthos_AlexNet_party0_metrics.csv',
    'Porthos_DenseNet121': '/home/mohebifarimah/logfiles/porthos_DenseNet_party0_metrics.csv',
    'Porthos_LeNet': '/home/mohebifarimah/logfiles/porthos_Lenet-large_party0_metrics.csv',
    'Porthos_ResNet50': '/home/mohebifarimah/logfiles/porthos_ResNet_party0_metrics.csv',
    'Porthos_ShuffleNetV2': '/home/mohebifarimah/logfiles/porthos_ShuffleNetV2_party0_metrics.csv',
    'Porthos_SqueezeNet_ImgNet': '/home/mohebifarimah/logfiles/porthos_SqueezeNetImgNet_party0_metrics.csv',
    'Porthos_SqueezeNet_CIFAR10': '/home/mohebifarimah/logfiles/porthos_SqueezeNetCIFAR10_party0_metrics.csv'
}

pm_files = {
    'AlexNet': '/home/mohebifarimah/logfilespm/SCI_AlexNet_client_metricspm.csv',
    'DenseNet121': '/home/mohebifarimah/logfilespm/SCI_DenseNet_client_metricspm.csv',
    'LeNet': '/home/mohebifarimah/logfilespm/SCI_Lenet-large_client_metricspm.csv',
    'ResNet50': '/home/mohebifarimah/logfilespm/SCI_ResNet_client_metricspm.csv',
    'ShuffleNetV2': '/home/mohebifarimah/logfilespm/SCI_ShuffleNetV2_client_metricspm.csv',
    'SqueezeNet_ImgNet': '/home/mohebifarimah/logfilespm/SCI_SqueezeNetImgNet_client_metricspm.csv',
    'SqueezeNet_CIFAR10': '/home/mohebifarimah/logfilespm/SCI_SqueezeNetCIFAR10_client_metricspm.csv',
    'Porthos_AlexNet': '/home/mohebifarimah/logfilespm/porthos_AlexNet_party0_metricspm.csv',
    'Porthos_DenseNet121': '/home/mohebifarimah/logfilespm/porthos_DenseNet_party0_metricspm.csv',
    'Porthos_LeNet': '/home/mohebifarimah/logfilespm/porthos_Lenet-large_party0_metricspm.csv',
    'Porthos_ResNet50': '/home/mohebifarimah/logfilespm/porthos_ResNet_party0_metricspm.csv',
    'Porthos_ShuffleNetV2': '/home/mohebifarimah/logfilespm/porthos_ShuffleNetV2_party0_metricspm.csv',
    'Porthos_SqueezeNet_ImgNet': '/home/mohebifarimah/logfilespm/porthos_SqueezeNetImgNet_party0_metricspm.csv',
    'Porthos_SqueezeNet_CIFAR10': '/home/mohebifarimah/logfilespm/porthos_SqueezeNetCIFAR10_party0_metricspm.csv'
}

# Define the operations to consider
operations = [
    'Total time in Conv (s)', 'Total time in MatMul (s)', 'Total time in BatchNorm (s)',
    'Total time in Truncation (s)', 'Total time in Relu (s)', 'Total time in MaxPool (s)',
    'Total time in AvgPool (s)', 'Total time in ArgMax (s)'
]

def extract_times(file_paths):
    combined_data = {}

    for model, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            for operation in operations:
                operation_data = df[df['Metric'] == operation]
                if not operation_data.empty:
                    combined_data.setdefault(operation, []).append(operation_data['Value'].values[0])
                else:
                    combined_data.setdefault(operation, []).append(None)  # No data available for this operation
        except Exception as e:
            print(f"Failed to read {path}: {e}")

    return combined_data

# Extract times for VM+Porthos VM and PM+Porthos PM
vm_data = extract_times(vm_files)
pm_data = extract_times(pm_files)

# Perform Wilcoxon Signed-Rank Test
for operation in operations:
    vm_operation = [x for x in vm_data[operation] if x is not None]
    pm_operation = [x for x in pm_data[operation] if x is not None]

    if len(vm_operation) > 1 and len(pm_operation) > 1:  # Need more than 1 value to perform the test
        try:
            stat, p_value = wilcoxon(vm_operation, pm_operation)
            print(f"Wilcoxon Test for {operation} - Statistic: {stat:.4f}, P-value: {p_value:.4f}")

            if p_value < 0.05:
                print(f"Significant difference in {operation} between VM (including Porthos VM) and PM (including Porthos PM) setups.\n")
            else:
                print(f"No significant difference in {operation} between VM (including Porthos VM) and PM (including Porthos PM) setups.\n")
        except ValueError as ve:
            print(f"Test not performed for {operation}: {ve}")
    else:
        print(f"Not enough data to perform test for {operation}.\n")
