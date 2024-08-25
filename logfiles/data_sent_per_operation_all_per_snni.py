import pandas as pd
import matplotlib.pyplot as plt

# Define paths for SNNI approaches including Porthos parties for each benchmark
paths = {
    'AlexNet': {
        'Cheetah': [
            'cheetah_alexnet_client_metrics.csv',
            'cheetah_alexnet_server_metrics.csv'
        ],
        'SCI_HE': [
            'SCI_HE_alexnet_client_metrics.csv',
            'SCI_HE_alexnet_server_metrics.csv'
        ],
        'SCI': [
            'SCI_AlexNet_client_metrics.csv',
            'SCI_AlexNet_server_metrics.csv'
        ],
        'Porthos': [
            'porthos_AlexNet_party0_metrics.csv',
            'porthos_AlexNet_party1_metrics.csv',
            'porthos_AlexNet_party2_metrics.csv'
        ]
    },
    # Additional benchmarks can be added here in a similar format
}

# List of data sent metrics of interest
expected_metrics = [
    'ArgMax data sent (MiB)', 'Avgpool data sent (MiB)', 'BatchNorm data sent (MiB)',
    'Conv data sent (MiB)', 'MatMul data sent (MiB)', 'Maxpool data sent (MiB)',
    'Relu data sent (MiB)', 'Truncation data sent (MiB)'
]

def aggregate_data(paths):
    all_data = []
    for benchmark, snni_paths in paths.items():
        for snni, files in snni_paths.items():
            df_list = []
            for file in files:
                try:
                    df = pd.read_csv(file)
                    df_filtered = df[df['Metric'].isin(expected_metrics)]
                    df_list.append(df_filtered)
                except FileNotFoundError:
                    print(f"File not found: {file}")
                    continue
            if df_list:
                combined_df = pd.concat(df_list)
                summed_df = combined_df.groupby('Metric')['Value'].sum().reset_index()
                summed_df['SNNI'] = snni
                summed_df['Benchmark'] = benchmark
                all_data.append(summed_df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def plot_data(df):
    if df.empty:
        print("No data to plot.")
        return

    for benchmark in df['Benchmark'].unique():
        benchmark_df = df[df['Benchmark'] == benchmark]
        pivot_df = benchmark_df.pivot(index='Metric', columns='SNNI', values='Value')
        ax = pivot_df.plot(kind='bar', figsize=(12, 8), logy=True, colormap='viridis')
        ax.set_ylabel('Total Data Sent (MiB)')
        ax.set_title(f'Total Data Sent in {benchmark}')
        plt.xticks(rotation=45)
        plt.legend(title='SNNI Approaches')
        plt.savefig(f'{benchmark}_total_data_sent.jpeg')
        plt.close()

# Process and plot data
aggregated_data = aggregate_data(paths)
plot_data(aggregated_data)
