import re
import csv
import sys

def extract_metric(pattern, text, default='0'):
    match = re.search(pattern, text)
    return match.group(1) if match else default

def process_porthos_log(log_file, output_csv):
    with open(log_file, 'r') as file:
        data = file.read()

    metrics = {
        "Number of CPU cores": extract_metric(r"Number of CPU cores: (\d+)", data),
        "CPU usage": extract_metric(r"CPU usage: ([\d.]+) %", data),
        "Estimated energy used": extract_metric(r"Estimated energy used: ([\d.]+) joules", data),
        "Elapsed wall time (s)": extract_metric(r"Elapsed wall time[:=]\s+([\d.]+)\s+seconds?", data),
        "Elapsed CPU time (s)": extract_metric(r"Elapsed CPU time[:=]\s+([\d.]+)\s+seconds?", data),
        "Total data sent (MiB)": extract_metric(r"Total data sent \(MiB\) = ([\d.]+)", data),
        "Total time in MatMul (s)": extract_metric(r"Total time in MatMul \(s\) = ([\d.]+)", data),
        "MatMul data sent (MiB)": extract_metric(r"MatMul data sent \(MiB\) = ([\d.]+)", data),
        "Total time in Relu (s)": extract_metric(r"Total time in Relu \(s\) = ([\d.]+)", data),
        "Relu data sent (MiB)": extract_metric(r"Relu data sent \(MiB\) = ([\d.]+)", data),
        "Total time in MaxPool (s)": extract_metric(r"Total time in MaxPool \(s\) = ([\d.]+)", data),
        "Maxpool data sent (MiB)": extract_metric(r"Maxpool data sent \(MiB\) = ([\d.]+)", data),
        "Total time in AvgPool (s)": extract_metric(r"Total time in AvgPool \(s\) = ([\d.]+)", data),
        "Avgpool data sent (MiB)": extract_metric(r"Avgpool data sent \(MiB\) = ([\d.]+)", data),
        "Total time in BatchNorm (s)": extract_metric(r"Total time in BatchNorm \(s\) = ([\d.]+)", data),
        "BatchNorm data sent (MiB)": extract_metric(r"BatchNorm data sent \(MiB\) = ([\d.]+)", data),
        "Total time in Conv (s)": extract_metric(r"Total time in Conv \(s\) = ([\d.]+)", data),
        "Conv data sent (MiB)": extract_metric(r"Conv data sent \(MiB\) = ([\d.]+)", data),
        "Total time in Truncation (s)": extract_metric(r"Total time in Truncation \(s\) = ([\d.]+)", data),
        "Truncation data sent (MiB)": extract_metric(r"Truncation data sent \(MiB\) = ([\d.]+)", data),
    }

    # Save metrics to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    print(f"Metrics have been saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_porthos_logs.py <log_file> <output_csv>")
        sys.exit(1)

    log_file = sys.argv[1]
    output_csv = sys.argv[2]

    process_porthos_log(log_file, output_csv)
