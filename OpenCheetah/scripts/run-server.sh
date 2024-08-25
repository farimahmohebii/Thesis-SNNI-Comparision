#!/bin/bash

# Source common functions and variables
. scripts/common.sh

# Function to check if a value is in a list
contains() {
  [[ $1 =~ (^|[[:space:]])$2($|[[:space:]]) ]]
}


# Validate the number of arguments
if [ ! $# -eq 2 ]; then
  echo -e "${RED}Please specify the network to run.${NC}"
  echo "Usage: run-server.sh [cheetah|SCI_HE] [sqnet/resnet50]"
  exit 1
fi

# Validate the first argument
if ! contains "cheetah SCI_HE" $1; then
  echo -e "Usage: run-server.sh ${RED}[cheetah|SCI_HE]${NC} [sqnet|resnet50|densenet121|alexnet|resnet101|resnet18|mnasnet|SqueezeNetCIFAR10|ffnn128]"
  exit 1
fi
echo "first part"
# Validate the second argument
if ! contains "sqnet resnet50 densenet121 alexnet resnet101 resnet18 mnasnet SqueezeNetCIFAR10 ffnn128 densenet169 shufflenetv2 lenet lenet-large sqnetacc" $2; then
  echo -e "Usage: run-server.sh [cheetah|SCI_HE] ${RED}[sqnet|resnet50|densenet121|alexnet|resnet101|resnet18|mnasnet|SqueezeNetCIFAR10|ffnn128]${NC}"
  exit 1
fi

# Create a data/ directory to store the Ferret output
mkdir -p data
ls -lh pretrained/$2_model_scale12.inp
echo -e "Running ${GREEN}build/bin/$2-$1${NC}, which might take a while...."

# Print debugging information
echo "Debugging Information:"
echo "Network: $1"
echo "Model: $2"
echo "Environment Variables:"
echo "FXP_SCALE: $FXP_SCALE"
echo "SS_BITLEN: $SS_BITLEN"
echo "NUM_THREADS: $NUM_THREADS"
echo "SERVER_PORT: $SERVER_PORT"

# Print meta information for debugging
#print_meta $1 $2

# Run the specified binary and ensure it signals when finished
cat pretrained/$2_model_scale12.inp | build/bin/$2-$1 r=1 k=$FXP_SCALE ell=$SS_BITLEN nt=$NUM_THREADS p=$SERVER_PORT 1>$1-$2_server.log

if [ $? -eq 0 ]; then
    echo "Server completed successfully."
    # You can create a file to signal that the server is done
    touch "/tmp/$1-$2-server-done"
else
    echo "Server encountered an error."
fi

echo -e "Computation done, check out the log file ${GREEN}$1-$2_server.log${NC}"
