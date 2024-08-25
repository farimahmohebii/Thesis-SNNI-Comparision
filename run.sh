#!/bin/bash

# Set the OpenCheetah and EzPC directories
OPEN_CHEETAH_DIR="$HOME/Thesis-SNNI-Comparision/OpenCheetah"
BUILD_DIR="$OPEN_CHEETAH_DIR/build/bin"  # Path to where the executables are located
EZPC_DIR="$HOME/Thesis-SNNI-Comparision/EzPC"  # Path to the EzPC directory

# Function to check and install dependencies

install_dependencies() {
    echo "Checking for required dependencies..."

    # List of dependencies
    dependencies=("openssl" "g++" "cmake" "git" "make" "libssl-dev" "libzstd-dev" "libomp-dev")

    for dep in "${dependencies[@]}"; do
        if ! dpkg -s "$dep" &>/dev/null; then
            echo "$dep is not installed. Installing..."
            sudo apt-get update
            sudo apt-get install -y "$dep"
        else
            echo "$dep is already installed."
        fi
    done

    # Check for the version of g++
    GPP_VERSION=$(g++ -dumpversion | cut -f1 -d.)
    if [ "$GPP_VERSION" -lt 8 ]; then
        echo "g++ version is less than 8.0. Please upgrade your g++ to version 8.0 or higher for better performance on AVX512."
        exit 1
    fi

    # Check for cmake version
    CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
    CMAKE_MAJOR_VERSION=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    CMAKE_MINOR_VERSION=$(echo "$CMAKE_VERSION" | cut -d. -f2)
    if [ "$CMAKE_MAJOR_VERSION" -lt 3 ] || { [ "$CMAKE_MAJOR_VERSION" -eq 3 ] && [ "$CMAKE_MINOR_VERSION" -lt 13 ]; }; then
        echo "cmake version is less than 3.13. Installing cmake..."
        sudo apt-get install -y cmake
    fi

    echo "All dependencies are installed."

    # Set OpenSSL root directory if not already set
    if [ -z "$OPENSSL_ROOT_DIR" ]; then
        OPENSSL_ROOT_DIR=$(openssl version -d | cut -d'"' -f2)
        export OPENSSL_ROOT_DIR
        echo "OPENSSL_ROOT_DIR is set to $OPENSSL_ROOT_DIR"
    fi

    # Set zstd directory if not already set
    if [ -z "$zstd_DIR" ]; then
        zstd_DIR=$(dpkg -L libzstd-dev | grep cmake | head -n 1 | sed 's|/libzstd.so||')
        export zstd_DIR
        echo "zstd_DIR is set to $zstd_DIR"
    fi

    # Install emp-tool library
    echo "Checking for emp-tool library..."

    if [ ! -d "/usr/local/include/emp-tool" ]; then
        echo "emp-tool is not installed. Installing emp-tool..."
        
        # Clone emp-tool repository
        git clone https://github.com/emp-toolkit/emp-tool.git
        
        # Build and install emp-tool
        cd emp-tool || exit
        mkdir -p build
        cd build || exit
        cmake -DOPENSSL_ROOT_DIR="$OPENSSL_ROOT_DIR" ..
        make
        sudo make install
        
        # Return to the original directory
        cd ../..

        echo "emp-tool library installed successfully."
    else
        echo "emp-tool library is already installed."
    fi

    # Install emp-ot library
    echo "Checking for emp-ot library..."

    if [ ! -d "~/Thesis-SNNI-Comparision/OpenCheetah/build/include/emp-ot" ]; then
        echo "emp-ot is not installed. Installing emp-ot..."
        
        # Clone emp-ot repository
        git clone https://github.com/emp-toolkit/emp-ot.git
        
        # Build and install emp-ot
        cd emp-ot || exit
        mkdir -p build
        cd build || exit
        
        # Configure CMake with proper prefix path for emp-tool and zstd
        cmake -DCMAKE_PREFIX_PATH=/usr/local -DOPENSSL_ROOT_DIR="$OPENSSL_ROOT_DIR" -Dzstd_DIR="$zstd_DIR" ..
        
        make
        sudo make install
        
        # Return to the original directory
        cd ../..
        
        echo "emp-ot library installed successfully."
    else
        echo "emp-ot library is already installed."
    fi

    echo "All libraries are installed and environment variables are set."
}


# Check and install dependencies at the beginning of the script
install_dependencies

# Function to check if the project is built
is_built() {
    # Check if the build/bin directory exists and contains executables
    if [ -d "$BUILD_DIR" ] && [ "$(ls -A $BUILD_DIR)" ]; then
        return 0
    else
        return 1
    fi
}

# Function to ensure the "cheetah" build is completed
ensure_cheetah_build() {
    if ! is_built; then
        echo "Cheetah is not built. Building the necessary components..."
        build_project
    else
        echo "Cheetah is already built."
    fi
}
# Function to build the project
build_project() {
    cd "$OPEN_CHEETAH_DIR" || exit
    bash scripts/build-deps.sh
    bash scripts/build.sh
    cd - || exit
}

activate_or_setup_venv() {
    cd "$EZPC_DIR" || { echo "Failed to navigate to EzPC directory."; exit 1; }

    # Check if the virtual environment exists
    if [ -d "mpc_venv" ]; then
        # Activate the virtual environment
        source mpc_venv/bin/activate
    else
        # Run the setup script and then activate the virtual environment
        ./setup_env_and_build.sh quick
        source mpc_venv/bin/activate
    fi
}

# Function to add a new benchmark
add_new_benchmark() {
    activate_or_setup_venv

    # Ask for benchmark type
    echo "Select the benchmark type:"
    echo "1) SCI"
    echo "2) Porthos"
    read -rp "Enter your choice (1/2): " benchmark_type_choice

    case $benchmark_type_choice in
        1)
            BENCHMARK_TYPE="SCI"
            ;;
        2)
            BENCHMARK_TYPE="PORTHOS"
            ;;
        *)
            echo "Invalid choice, please try again."
            return
            ;;
    esac

    # Ask for benchmark name
    read -rp "Enter the benchmark name: " BENCHMARK_NAME

    # Create a new folder in the Athos/Networks directory
    BENCHMARK_DIR="$EZPC_DIR/Athos/Networks/$BENCHMARK_NAME"
    mkdir -p "$BENCHMARK_DIR" || { echo "Failed to create benchmark directory."; return; }
    echo "Created benchmark directory at $BENCHMARK_DIR"

    # Ask for model type
    echo "Select the model type:"
    echo "1) ONNX"
    echo "2) TensorFlow"
    read -rp "Enter your choice (1/2): " model_type_choice

    if [[ $model_type_choice -eq 1 ]]; then
        MODEL_TYPE="ONNX"
        read -rp "Enter the link to the ONNX model: " model_link
        wget -O "$BENCHMARK_DIR/model.onnx" "$model_link" || { echo "Failed to download model."; return; }
        echo "Downloaded model to $BENCHMARK_DIR/model.onnx"
    else
        echo "Only ONNX model support is implemented."
        return
    fi

    # Download the input image
    read -rp "Enter the link to the input image: " image_link
    wget -O "$BENCHMARK_DIR/input.jpg" "$image_link" || { echo "Failed to download image."; return; }
    echo "Downloaded input image to $BENCHMARK_DIR/input.jpg"
    
    # Get the directory where the .sh file is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Define the paths for image preprocessing
    IMAGE_PATH="$BENCHMARK_DIR/input.jpg"
    OUTPUT_PATH="$BENCHMARK_DIR/preprocessed_image.npy"
    
    # Run the preprocessing script using the path where the .sh file is located
    python "$SCRIPT_DIR/preprocess_image.py" "$IMAGE_PATH" "$OUTPUT_PATH"
    
    # Ask for output tensor names
    read -rp "Enter the output tensor names (comma-separated): " output_tensors

    # Handle SCI-specific configuration and setup
    if [[ "$BENCHMARK_TYPE" == "SCI" ]]; then
        config_file="$BENCHMARK_DIR/config.json"
        MODEL_PATH="$BENCHMARK_DIR/model.onnx"

        # Create the config.json with the full model path for SCI
        cat <<EOL > "$BENCHMARK_DIR/config.json"
{
  "model_name": "$MODEL_PATH",
  "output_tensors": ["$output_tensors"],
  "target": "SCI", 
  "scale": 12,
  "backend": "OT",
  "bitlength": 64
}
EOL
        echo "Config file created at $config_file"

        # Compile the ONNX model
        python ~/Thesis-SNNI-Comparision/EzPC/Athos/CompileONNXGraph.py --config "$BENCHMARK_DIR/config.json" --role server
        
        # After compiling the ONNX graph, run the conversion of numpy array to fixed-point
        python ~/Thesis-SNNI-Comparision/EzPC/Athos/CompilerScripts/convert_np_to_fixedpt.py --inp "$BENCHMARK_DIR/preprocessed_image.npy" --config "$BENCHMARK_DIR/config.json"

        echo "Fixed-point conversion completed and saved in the same directory."

        # Full path to the fixed-point files generated after conversion
        WEIGHTS_FILE="$BENCHMARK_DIR/model_input_weights_fixedpt_scale_12.inp"
        IMAGE_FILE="$BENCHMARK_DIR/preprocessed_image_fixedpt_scale_12.inp"

        # Rename the weights file
        if [ -f "$WEIGHTS_FILE" ]; then
            mv "$WEIGHTS_FILE" "$BENCHMARK_DIR/model_weights_scale_12.inp"
        else
            echo "File $WEIGHTS_FILE not found!"
        fi


        # Rename the preprocessed image file
        if [ -f "$IMAGE_FILE" ]; then
            mv "$IMAGE_FILE" "$BENCHMARK_DIR/model_input_scale_12.inp"
        else
            echo "File $IMAGE_FILE not found!"
        fi

        # Rename the SCI executable
        if [[ -f "$BENCHMARK_DIR/model_SCI_OT.out" ]]; then
            mv "$BENCHMARK_DIR/model_SCI_OT.out" "$BENCHMARK_DIR/${BENCHMARK_NAME}_SCI_OT.out"
            echo "Renamed model_SCI_OT.out to ${BENCHMARK_NAME}_SCI_OT.out"
        else
            echo "SCI executable model_SCI_OT.out not found in $BENCHMARK_DIR."
        fi

        echo "SCI benchmark setup is complete."
    
    # Handle Porthos-specific configuration and setup
    elif [[ "$BENCHMARK_TYPE" == "PORTHOS" ]]; then
        config_file="$BENCHMARK_DIR/config.json"
        MODEL_PATH="$BENCHMARK_DIR/model.onnx"

        # Create the config.json with the full model path for Porthos
        cat <<EOL > "$BENCHMARK_DIR/config.json"
{
  "model_name": "$MODEL_PATH",
  "output_tensors": ["$output_tensors"],
  "target": "PORTHOS", 
  "scale": 12,
  "backend": "OT",
  "bitlength": 64
}
EOL
        echo "Config file created at $config_file"

        # Compile the ONNX model
        python ~/Thesis-SNNI-Comparision/EzPC/Athos/CompileONNXGraph.py --config "$BENCHMARK_DIR/config.json" --role server
        
        # After compiling the ONNX graph, run the conversion of numpy array to fixed-point
        python ~/Thesis-SNNI-Comparision/EzPC/Athos/CompilerScripts/convert_np_to_fixedpt.py --inp "$BENCHMARK_DIR/preprocessed_image.npy" --config "$BENCHMARK_DIR/config.json"

        echo "Fixed-point conversion completed and saved in the same directory."

        # Full path to the fixed-point files generated after conversion
        WEIGHTS_FILE="$BENCHMARK_DIR/model_input_weights_fixedpt_scale_12.inp"
        IMAGE_FILE="$BENCHMARK_DIR/preprocessed_image_fixedpt_scale_12.inp"

        # Rename the weights file
        if [ -f "$WEIGHTS_FILE" ]; then
            mv "$WEIGHTS_FILE" "$BENCHMARK_DIR/model_weights_scale_12.inp"
        else
            echo "File $WEIGHTS_FILE not found!"
        fi

        # Rename the preprocessed image file
        if [ -f "$IMAGE_FILE" ]; then
            mv "$IMAGE_FILE" "$BENCHMARK_DIR/model_input_scale_12.inp"
        else
            echo "File $IMAGE_FILE not found!"
        fi

        # Rename the Porthos executable
        if [[ -f "$BENCHMARK_DIR/model_PORTHOS.out" ]]; then
            mv "$BENCHMARK_DIR/model_PORTHOS.out" "$BENCHMARK_DIR/${BENCHMARK_NAME}_PORTHOS.out"
            echo "Renamed model_PORTHOS.out to ${BENCHMARK_NAME}_PORTHOS.out"
        else
            echo "Porthos executable model_PORTHOS.out not found in $BENCHMARK_DIR."
        fi

        echo "Porthos benchmark setup is complete."
    else
        echo "Unsupported benchmark type."
        return
    fi

    deactivate
    prompt_continue
}




# Function to list available benchmarks with numbering
list_benchmarks() {
    local choice=$1
    echo "Looking for benchmarks with $choice..."
    if [ -d "$BUILD_DIR" ]; then
        local benchmarks=($(ls "$BUILD_DIR" | grep "\-$choice" | sed "s/-$choice//"))  # Remove the '-choice' part
        local index=1
        echo "Available benchmarks:"
        for benchmark in "${benchmarks[@]}"; do
            printf "%d) %s\n" "$index" "$benchmark"  # Print each benchmark on a new line with its index
            index=$((index + 1))
        done
    else
        echo "No benchmarks found. Please ensure the build process is completed correctly."
    fi
}

# Function to run the selected benchmark
run_benchmark() {
    local choice=$1
    local benchmark=$2
    local benchmark_name=${benchmark%-*}
    local open_cheetah_dir="$HOME/OpenCheetah"
    local log_path="$open_cheetah_dir/logs"

    echo "Starting benchmark: $benchmark_name under mode: $choice"

    # Ensure log directory exists
    mkdir -p "$log_path"

    # Navigate to the correct directory
    cd "$open_cheetah_dir" || exit 1


    # Run server and client scripts in the background, using proper naming for logs
    bash scripts/run-server.sh $choice $benchmark_name > "$log_path/${choice}-${benchmark_name}_server.log" 2>&1 &
    server_pid=$!
    echo "Server started for $benchmark_name, logging to ${choice}-${benchmark_name}_server.log, PID $server_pid"

    bash scripts/run-client.sh $choice $benchmark_name > "$log_path/${choice}-${benchmark_name}_client.log" 2>&1 &
    client_pid=$!
    echo "Client started for $benchmark_name, logging to ${choice}-${benchmark_name}_client.log, PID $client_pid"
    
    
    wait $server_pid
    echo "Server script has completed."
    wait $client_pid
    echo "Client script has completed."

    
    echo "Both scripts have completed. Check the log files for output details."

    python3 ../process_logs_client.py "${choice}-${benchmark_name}_client.log" "../logfiles/${choice}_${benchmark_name}_client_metrics.csv"
    python3 ../process_logs_server.py "${choice}-${benchmark_name}_server.log" "../logfiles/${choice}_${benchmark_name}_server_metrics.csv"
    
    echo "Metrics (client) have been saved to "${choice}_${benchmark_name}_client_metrics.csv""
    echo "Metrics (server) have been saved to "$log_path/${choice}_${benchmark_name}_server_metrics.csv""

    # Return to the main menu
    prompt_continue
}

# Function to list all CSV files

list_all_csv_files() {
    # Gather CSV files from OpenCheetah logs
    csv_files=("$OPEN_CHEETAH_DIR/logs"/*.csv)

    # Gather CSV files from EzPC/Athos/Networks
    for dir in "$EZPC_DIR/Athos/Networks"/*; do
        if [ -d "$dir" ]; then
            csv_files+=("$dir"/*.csv)
        fi
    done

    echo "Available CSV files:"
    local index=1
    for file in "${csv_files[@]}"; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            if [[ $filename =~ ^(SCI_HE|cheetah|porthos|SCI)_[^_]+_metrics\.csv$ ]]; then
                echo "$index) $filename"
                index=$((index + 1))
            fi
        fi
    done
}

# Function to select a CSV file
select_csv_file() {
    local selection
    while true; do
        read -p "Enter the number of the CSV file to select: " selection
        if [[ -n "${csv_file_map[$selection]}" ]]; then
            echo "${csv_file_map[$selection]}"
            return
        else
            echo "Invalid selection, please try again."
        fi
    done
}

# Function to run SCI mode
run_sci_mode() {
    echo "Running SCI mode..."
    cd "$EZPC_DIR" || { echo "Failed to navigate to EzPC directory."; exit 1; }

    # Activate the virtual environment
    source mpc_venv/bin/activate

    # Navigate to Athos/Networks and list directories
    cd Athos/Networks || { echo "Failed to navigate to Athos/Networks directory."; exit 1; }
    echo "Available folders in Athos/Networks:"
    folders=(*/)
    select folder in "${folders[@]%/}"; do
        if [[ -n $folder ]]; then
            echo "You selected: $folder"
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done

  # Navigate to the selected folder
    cd "$folder" || { echo "Failed to navigate to selected folder: $folder"; deactivate; exit 1; }

    # Ask whether to run locally or remotely
    echo "Do you want to run this locally or remotely?"
    echo "1) Locally"
    echo "2) Remotely"
    read -rp "Enter your choice (1/2): " run_choice

    case $run_choice in
        1)  # Run locally
            echo "Running SCI benchmark locally..."

            # Measure initial energy consumption
            initial_energy=$(sudo cat /sys/class/powercap/intel-rapl:0/energy_uj)
            echo "Initial energy reading: $initial_energy µJ"

            # Run commands in the background
            ./"${folder}_SCI_OT.out" r=1 < model_weights_scale_12.inp > "SCI-${folder}_server.log" &
            sci_r1_pid=$!
            echo "Running ${folder}_SCI_OT.out r=1 in background, PID $sci_r1_pid"

            ./"${folder}_SCI_OT.out" r=2 < model_input_scale_12.inp > "SCI-${folder}_client.log" &
            sci_r2_pid=$!
            echo "Running ${folder}_SCI_OT.out r=2 in background, PID $sci_r2_pid"

            # Wait for both processes to finish
            wait $sci_r1_pid
            echo "SCI r=1 process completed."
            wait $sci_r2_pid
            echo "SCI r=2 process completed."

            # Measure final energy consumption
            final_energy=$(sudo cat /sys/class/powercap/intel-rapl:0/energy_uj)
            echo "Final energy reading: $final_energy µJ"

            # Calculate energy usage in joules
            energy_used=$(echo "scale=6; ($final_energy - $initial_energy) / 1000000" | bc)
            echo "Total energy used: $energy_used J"

            # Generate CSV
            python3 ../../../../process_logs_client.py "SCI-${folder}_client.log" "../../../../logfiles/SCI_${folder}_client_metrics.csv"
            python3 ../../../../process_logs_server.py "SCI-${folder}_server.log" "../../../../logfiles/SCI_${folder}_server_metrics.csv"

            echo "Metrics (client) have been saved to SCI_${folder}_client_metrics.csv"
            echo "Metrics (server) have been saved to SCI_${folder}_server_metrics.csv"
            ;;

        2)  # Run remotely
            echo "Running SCI benchmark remotely..."
            echo "Do you want to run as a client or a server?"
            echo "1) Server"
            echo "2) Client"
            read -rp "Enter your choice (1/2): " remote_choice

            if [[ "$remote_choice" == "1" ]]; then
                read -rp "Enter the port number to run on: " port_number

                # Run server command
                ./"${folder}_SCI_OT.out" r=1 p=$port_number < model_weights_scale_12.inp > "SCI-${folder}_server.log" &
                sci_r1_pid=$!
                echo "Running ${folder}_SCI_OT.out r=1 on port $port_number in background, PID $sci_r1_pid"

                # Wait for the process to finish
                wait $sci_r1_pid
                echo "SCI server process completed."

                # Generate server metrics CSV
                python3 ../../../../process_logs_server.py "SCI-${folder}_server.log" "../../../../logfiles/SCI_${folder}_server_metrics.csv"
                echo "Server metrics have been saved to SCI_${folder}_server_metrics.csv"

            elif [[ "$remote_choice" == "2" ]]; then
                read -rp "Enter the port number to connect to: " port_number
                read -rp "Enter the IP address of the server: " server_ip

                # Run client command
                ./"${folder}_SCI_OT.out" r=2 p=$port_number IP=$server_ip < model_input_scale_12.inp > "SCI-${folder}_client.log" &
                sci_r2_pid=$!
                echo "Running ${folder}_SCI_OT.out r=2 connecting to $server_ip on port $port_number in background, PID $sci_r2_pid"

                # Wait for the process to finish
                wait $sci_r2_pid
                echo "SCI client process completed."

                # Generate client metrics CSV
                python3 ../../../../process_logs_client.py "SCI-${folder}_client.log" "../../../../logfiles/SCI_${folder}_client_metrics.csv"
                echo "Client metrics have been saved to SCI_${folder}_client_metrics.csv"
            else
                echo "Invalid choice, please select either 1 (Server) or 2 (Client)."
                deactivate
                return
            fi
            ;;
        
        *)
            echo "Invalid choice. Please select either 1 (Locally) or 2 (Remotely)."
            deactivate
            return
            ;;
    esac

    # Deactivate the virtual environment
    deactivate

    # Return to the main menu
    prompt_continue
}


# Function to run Porthos mode
run_porthos_mode() {
    echo "Running Porthos mode..."
    cd "$EZPC_DIR" || { echo "Failed to navigate to EzPC directory."; exit 1; }

    # Check if the virtual environment exists
    if [ -d "mpc_venv" ]; then
        # Activate the virtual environment
        source mpc_venv/bin/activate
    else
        # Run the setup script and then activate the virtual environment
        ./setup_env_and_build.sh quick
        source mpc_venv/bin/activate
    fi

    # Navigate to Athos/Networks and list directories
    cd Athos/Networks || { echo "Failed to navigate to Athos/Networks directory."; exit 1; }
    echo "Available benchmarks"
    folders=(*/)
   # local index=1
    #for folder in "${folders[@]%/}"; do
     #   printf "%d) %s\n" "$index" "$folder"
      #  index=$((index + 1))
   # done   

 select folder in "${folders[@]%/}"; do
        if [[ -n $folder ]]; then
            echo "You selected: $folder"
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done

    # Navigate to the selected folder
    cd "$folder" || { echo "Failed to navigate to selected folder: $folder"; deactivate; exit 1; }

    # Run the Porthos commands in the background and log the outputs
    ./"${folder}_PORTHOS.out" 0 ../../../Porthos/files/addresses ../../../Porthos/files/keys < model_input_scale_12.inp > "${folder}_output_porthos_0.txt" &
    porthos_r0_pid=$!
    echo "Running ${folder}_PORTHOS.out r=0 in background, PID $porthos_r0_pid"

    ./"${folder}_PORTHOS.out" 1 ../../../Porthos/files/addresses ../../../Porthos/files/keys < model_weights_scale_12.inp > "${folder}_output_porthos_1.txt" &
    porthos_r1_pid=$!
    echo "Running ${folder}_PORTHOS.out r=1 in background, PID $porthos_r1_pid"

    ./"${folder}_PORTHOS.out" 2 ../../../Porthos/files/addresses ../../../Porthos/files/keys > "${folder}_output_porthos_2.txt" &
    porthos_r2_pid=$!
    echo "Running ${folder}_PORTHOS.out r=2 in background, PID $porthos_r2_pid"


   # Monitor CPU usage and other metrics with perf for the first command
  # sudo perf stat -p $porthos_r0_pid -e task-clock,cpu-clock,context-switches,cpu-migrations,page-faults,cycles,instructions -o party0_perf_output.txt &

   # Monitor CPU usage and other metrics with perf for the second command
  # sudo perf stat -p $porthos_r1_pid -e task-clock,cpu-clock,context-switches,cpu-migrations,page-faults,cycles,instructions -o party1_perf_output.txt &
   
   # Monitor CPU usage and other metrics with perf for the third command
  # sudo perf stat -p $porthos_r2_pid -e task-clock,cpu-clock,context-switches,cpu-migrations,page-faults,cycles,instructions -o party2_perf_output.txt &


    # Wait for all Porthos processes to finish
    wait $porthos_r0_pid
    echo "PORTHOS r=0 process completed."
    wait $porthos_r1_pid
    echo "PORTHOS r=1 process completed."
    wait $porthos_r2_pid
    echo "PORTHOS r=2 process completed."
 

    kill $porthos_r0_pid
    kill $porthos_r1_pid
    kill $porthos_r2_pid

    # Process the log file and generate CSV
    python3 ../../../../process_logs_serverp.py "${folder}_output_porthos_0.txt" "../../../../logfiles/porthos_${folder}_party0_metrics.csv"

    echo "Porthos metrics CSV file generated: porthos_${folder}_party0_metrics.csv"

    python3 ../../../../process_logs_serverp.py "${folder}_output_porthos_1.txt" "../../../../logfiles/porthos_${folder}_party1_metrics.csv"
    
    echo "Porthos metrics CSV file generated: porthos_${folder}_party1_metrics.csv"

    python3 ../../../../process_logs_serverp.py "${folder}_output_porthos_2.txt" "../../../../logfiles/porthos_${folder}_party2_metrics.csv"
    
    echo "Porthos metrics CSV file generated: porthos_${folder}_party2_metrics.csv"


    # Deactivate the virtual environment
    deactivate

    # Return to the main menu
    prompt_continue
}

# Function to prompt user to continue or exit
prompt_continue() {
    while true; do
        read -p "Do you want to go back to the main menu? (yes/no): " yn
        case $yn in
            [Yy]* ) 
                echo "Returning to main menu..."
                break;;
            [Nn]* ) 
                echo "Exiting..."
                exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to compare CSV files


compare_csv_files() {
    list_all_csv_files

    # Ask the user how many CSV files they want to compare
    while true; do
        read -p "How many benchmark outputs do you want to compare? (2-7): " num_files
        if [[ "$num_files" =~ ^[2-7]$ ]]; then
            break
        else
            echo "Please enter a number between 2 and 7."
        fi
    done

 selected_files=()

    for ((i=1; i<=num_files; i++)); do
        echo "Select output files of different benchmarks $i:"
        csv=$(select_csv_file)
        if [[ ! -f "$csv" ]]; then
            echo "File not found: $csv"
            return 1
        fi
        selected_files+=("$csv")
    done

    echo "Selected CSV files:"
    printf "%s\n" "${selected_files[@]}"

    # Run the comparison script with the selected files
    python3 ~/Thesis-SNNI-Comparision/compare_csv.py "${selected_files[@]}"
}

# Main loop
while true; do
    echo "Choose a mode: (1) cheetah (2) SCI_HE (3) SCI (4) Porthos (5) Compare Result of Applying SNNI approaches on Benchmarks (6) Add New Benchmark (7) Exit"
    read -rp "Enter your choice: " mode_choice
    
    case $mode_choice in
        1 | "cheetah" | "Cheetah")
            MODE="cheetah"
            ensure_cheetah_build
            ;;
        2 | "SCI_HE" | "sci_he" | "SCI HE" | "sci he")
            MODE="SCI_HE"
            ;;
        3 | "SCI" | "sci")
            run_sci_mode
            continue  # Skip further processing and return to menu
            ;;
        4 | "Porthos" | "porthos")
            run_porthos_mode
            continue  # Skip further processing and return to menu
            ;;
        5 | "compare" | "Compare")
            compare_csv_files
            continue  # Skip further processing and return to menu
            ;;
        6 | "add" | "Add New Benchmark")
            add_new_benchmark
            continue  # Skip further processing and return to menu
            ;;
        7 | "exit" | "Exit" | "quit" | "Quit")
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice, please try again."
            continue
            ;;
    esac
    
    if ! is_built; then
        echo "Building the project..."
        build_project
    else
        echo "Project is already built."
    fi
    
    echo "Available benchmarks for $MODE:"
    list_benchmarks "$MODE"

    # Get user selection for benchmark
    read -rp "Enter the number of the benchmark you want to run: " benchmark_index
    benchmark_choice=$(ls "$BUILD_DIR" | grep "\-$MODE" | sed -n "${benchmark_index}p")

    if [[ -z "$benchmark_choice" ]]; then
        echo "Invalid selection, please try again."
        continue
    fi
    
    run_benchmark "$MODE" "$benchmark_choice"
done

