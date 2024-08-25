import sys
import re
import csv


def extract_metric(pattern, text, default="0"):
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
        with open(log_filename, "r") as file:
            data = file.read()
            metrics = {
                "Number of CPU cores": extract_metric(
                    r"Number of CPU cores: (\d+)", data
                ),
                "Elapsed wall time (s)": extract_metric(
                    r"Elapsed wall time: ([\d.]+) seconds", data
                ),
                "Elapsed CPU time (s)": extract_metric(
                    r"Elapsed CPU time: ([\d.]+) seconds", data
                ),
                "CPU usage (%)": extract_metric(r"CPU usage: ([\d.]+) %", data),
                "Total time in Conv (s)": extract_metric(
                    r"Total time in Conv = ([\d.]+) seconds", data
                ),
                "Total time in MatMul (s)": extract_metric(
                    r"Total time in MatMul = ([\d.]+) seconds", data
                ),
                "Total time in BatchNorm (s)": extract_metric(
                    r"Total time in BatchNorm = ([\d.]+) seconds", data
                ),
                "Total time in Truncation (s)": extract_metric(
                    r"Total time in Truncation = ([\d.]+) seconds", data
                ),
                "Total time in Relu (s)": extract_metric(
                    r"Total time in Relu = ([\d.]+) seconds", data
                ),
                "Total time in MaxPool (s)": extract_metric(
                    r"Total time in MaxPool = ([\d.]+) seconds", data
                ),
                "Total time in AvgPool (s)": extract_metric(
                    r"Total time in AvgPool = ([\d.]+) seconds", data
                ),
                "Total time in ArgMax (s)": extract_metric(
                    r"Total time in ArgMax = ([\d.]+) seconds", data
                ),
                "Total time in MatAdd (s)": extract_metric(
                    r"Total time in MatAdd = ([\d.]+) seconds", data
                ),
                "Total time in MatAddBroadCast (s)": extract_metric(
                    r"Total time in MatAddBroadCast = ([\d.]+) seconds", data
                ),
                "Total time in MulCir (s)": extract_metric(
                    r"Total time in MulCir = ([\d.]+) seconds", data
                ),
                "Total time in ScalarMul (s)": extract_metric(
                    r"Total time in ScalarMul = ([\d.]+) seconds", data
                ),
                "Total time in Sigmoid (s)": extract_metric(
                    r"Total time in Sigmoid = ([\d.]+) seconds", data
                ),
                "Total time in Tanh (s)": extract_metric(
                    r"Total time in Tanh = ([\d.]+) seconds", data
                ),
                "Total time in Sqrt (s)": extract_metric(
                    r"Total time in Sqrt = ([\d.]+) seconds", data
                ),
                "Total time in NormaliseL2 (s)": extract_metric(
                    r"Total time in NormaliseL2 = ([\d.]+) seconds", data
                ),
                "Total data sent (MiB)": extract_metric(
                    r"Total data sent = ([\d.]+) MiB", data
                ),
                "Estimated energy used (J)": extract_metric(
                    r"Estimated energy used: ([\d.e+-]+) joules", data
                ),
                "Total comm rounds": extract_metric(r"Number of rounds = (\d+)", data),
                "Conv data sent (MiB)": extract_metric(
                    r"Conv data sent = ([\d.]+) MiB", data
                ),
                "MatMul data sent (MiB)": extract_metric(
                    r"MatMul data sent = ([\d.]+) MiB", data
                ),
                "BatchNorm data sent (MiB)": extract_metric(
                    r"BatchNorm data sent = ([\d.]+) MiB", data
                ),
                "Truncation data sent (MiB)": extract_metric(
                    r"Truncation data sent = ([\d.]+) MiB", data
                ),
                "Relu data sent (MiB)": extract_metric(
                    r"Relu data sent = ([\d.]+) MiB", data
                ),
                "Maxpool data sent (MiB)": extract_metric(
                    r"Maxpool data sent = ([\d.]+) MiB", data
                ),
                "Avgpool data sent (MiB)": extract_metric(
                    r"Avgpool data sent = ([\d.]+) MiB", data
                ),
                "ArgMax data sent (MiB)": extract_metric(
                    r"ArgMax data sent = ([\d.]+) MiB", data
                ),
                "MatAdd data sent (MiB)": extract_metric(
                    r"MatAdd data sent = ([\d.]+) MiB", data
                ),
                "MatAddBroadCast data sent (MiB)": extract_metric(
                    r"MatAddBroadCast data sent = ([\d.]+) MiB", data
                ),
                "MulCir data sent (MiB)": extract_metric(
                    r"MulCir data sent = ([\d.]+) MiB", data
                ),
                "Sigmoid data sent (MiB)": extract_metric(
                    r"Sigmoid data sent = ([\d.]+) MiB", data
                ),
                "Tanh data sent (MiB)": extract_metric(
                    r"Tanh data sent = ([\d.]+) MiB", data
                ),
                "Sqrt data sent (MiB)": extract_metric(
                    r"Sqrt data sent = ([\d.]+) MiB", data
                ),
                "NormaliseL2 data sent (MiB)": extract_metric(
                    r"NormaliseL2 data sent = ([\d.]+) MiB", data
                ),
            }

            # Save the metrics to a CSV file
            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Value"])
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
