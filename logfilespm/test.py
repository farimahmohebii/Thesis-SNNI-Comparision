import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# Define paths to CSV files for VM (including Porthos VM) and PM (including Porthos PM) setups
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

# Function to extract elapsed wall time from CSV files
def extract_elapsed_time(file_paths):
    elapsed_times = {}
    for benchmark, file_path in file_paths.items():
        try:
            df = pd.read_csv(file_path)
            elapsed_time = df.loc[df['Metric'] == 'Elapsed wall time (s)', 'Value'].values
            if len(elapsed_time) > 0:
                elapsed_times[benchmark] = elapsed_time[0]
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    return elapsed_times

# Extract elapsed times for VM and PM setups (including Porthos data)
vm_elapsed_times = extract_elapsed_time(vm_files)
pm_elapsed_times = extract_elapsed_time(pm_files)

# Check if data is available for both setups
if not vm_elapsed_times or not pm_elapsed_times:
    print("No data available for one or more setups.")
else:
    # Prepare data for Wilcoxon Signed-Rank Test
    benchmarks = list(vm_elapsed_times.keys())
    vm_times = [vm_elapsed_times[benchmark] for benchmark in benchmarks]
    pm_times = [pm_elapsed_times[benchmark] for benchmark in benchmarks]

    # Perform Wilcoxon Signed-Rank Test between VM and PM setups
    try:
        stat_vm_pm, p_value_vm_pm = wilcoxon(vm_times, pm_times)
        print(f'Wilcoxon Test (VM+Porthos VM vs PM+Porthos PM) - Statistic: {stat_vm_pm:.4f}, P-value: {p_value_vm_pm:.4f}')

        # Interpretation
        if p_value_vm_pm < 0.05:
            print("Significant difference in elapsed wall time between VM (including Porthos) and PM (including Porthos) setups.")
        else:
            print("No significant difference in elapsed wall time between VM (including Porthos) and PM (including Porthos) setups.")

    except ValueError as e:
        print(f"Could not perform the Wilcoxon test: {e}")

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.bar(benchmarks, vm_times, color='blue', alpha=0.6, label='VM + Porthos VM Elapsed Time')
    plt.bar(benchmarks, pm_times, color='red', alpha=0.6, label='PM + Porthos PM Elapsed Time')
    plt.ylabel('Elapsed Wall Time (s)')
    plt.title('Comparison of Elapsed Wall Time Across Setups (VM+Porthos VM vs PM+Porthos PM)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('elapsed_wall_time_comparison_across_combined_setups.jpeg', format='jpeg')
    plt.close()
