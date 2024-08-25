import sys
import re
import csv

def extract_metric(pattern, text, default='0'):
    """
    Extracts a metric from the provided text using regex.
    If the pattern is not found, returns a default value.
    """
    match = re.search(pattern, text)
    return match.group(1) if match else default

def process_log_to_csv(log_filename, csv_filename):
    """
    Processes a log file to extract metrics and saves them to a CSV file.
    """
    try:
        with open(log_filename, 'r') as file:
            data = file.read()
            metrics = {
                "Number of CPU cores": extract_metric(r"Number of CPU cores: (\d+)", data),
                "Average CPU usage (%)": extract_metric(r"Average CPU usage across all segments: ([\d.]+)", data),
                "Peak CPU usage (%)": extract_metric(r"Peak CPU usage during computation: ([\d.]+)", data),
                "Estimated energy used (J)": extract_metric(r"Estimated energy used: ([\d.e+-]+) joules", data),
                "Elapsed wall time (s)": extract_metric(r"Elapsed wall time: ([\d.]+) seconds", data),
                "Elapsed CPU time (s)": extract_metric(r"Elapsed CPU time: ([\d.]+) seconds", data),
                "Total time taken (ms)": extract_metric(r"Total time taken \(ms\) = ([\d.]+)", data),
                "Total data sent (MiB)": extract_metric(r"Total data sent \(MiB\) = ([\d.]+)", data),
                "Total comm (sent+received) (MiB)": extract_metric(r"Total comm \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "Conv data (sent+received) (MiB)": extract_metric(r"Conv data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "MatMul data (sent+received) (MiB)": extract_metric(r"MatMul data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "BatchNorm data (sent+received) (MiB)": extract_metric(r"BatchNorm data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "Truncation data (sent+received) (MiB)": extract_metric(r"Truncation data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "Relu data (sent+received) (MiB)": extract_metric(r"Relu data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "Maxpool data (sent+received) (MiB)": extract_metric(r"Maxpool data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "Avgpool data (sent+received) (MiB)": extract_metric(r"Avgpool data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "ArgMax data (sent+received) (MiB)": extract_metric(r"ArgMax data \(sent\+received\) \(MiB\) = ([\d.]+)", data),
                "Total time in Conv (s)": extract_metric(r"Total time in Conv \(s\) = ([\d.]+)", data),
                "Total time in MatMul (s)": extract_metric(r"Total time in MatMul \(s\) = ([\d.]+)", data),
                "Total time in BatchNorm (s)": extract_metric(r"Total time in BatchNorm \(s\) = ([\d.]+)", data),
                "Total time in Truncation (s)": extract_metric(r"Total time in Truncation \(s\) = ([\d.]+)", data),
                "Total time in Relu (s)": extract_metric(r"Total time in Relu \(s\) = ([\d.]+)", data),
                "Total time in MaxPool (s)": extract_metric(r"Total time in MaxPool \(s\) = ([\d.]+)", data),
                "Total time in AvgPool (s)": extract_metric(r"Total time in AvgPool \(s\) = ([\d.]+)", data),
                "Total time in ArgMax (s)": extract_metric(r"Total time in ArgMax \(s\) = ([\d.]+)", data),
            }

            # Save the metrics to a CSV file
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value'])
                for key, value in metrics.items():
                    writer.writerow([key, value])

            print(f"Metrics have been saved to {csv_filename}")

    except Exception as e:
        print(f"Error processing the log file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_logs.py <log_filename> <csv_filename>")
    else:
        process_log_to_csv(sys.argv[1], sys.argv[2])
