. scripts/common.sh

if [ ! $# -eq 2 ]
then
  echo -e "${RED}Please specify the network to run.${NC}"
  echo "Usage: run-client.sh [cheetah|SCI_HE] [sqnet/resnet50]"
  exit 1
else
  if ! contains "cheetah SCI_HE" $1; then
    echo -e "Usage: run-client.sh ${RED}[cheetah|SCI_HE]${NC} [sqnet|resnet50|densenet121|alexnet|resnet101|resnet18|mnasnet|SqueezeNetCIFAR10|ffnn128]"
    exit 1
  fi

  if ! contains "sqnet resnet50 densenet121 alexnet resnet101 resnet18 mnasnet SqueezeNetCIFAR10 ffnn128 densenet169 shufflenetv2 lenet lenet-large sqnetacc" $2; then
    echo -e "Usage: run-client.sh [cheetah|SCI_HE] ${RED}[sqnet|resnet50|densenet121|alexnet|resnet101|resnet18|mnasnet|SqueezeNetCIFAR10|ffnn128]${NC}"
    exit 1
  fi
  
  # create a data/ to store the Ferret output
  mkdir -p data
  echo -e "Running ${GREEN}build/bin/$2-$1${NC}, which might take a while...."

  # Execute the client binary and redirect its output to a log file
  cat pretrained/$2_input_scale12_pred*.inp | build/bin/$2-$1 r=2 k=$FXP_SCALE ell=$SS_BITLEN nt=$NUM_THREADS ip=$SERVER_IP p=$SERVER_PORT 1>$1-$2_client.log

  # Check if the operation was successful
  if [ $? -eq 0 ]; then
    echo "Client completed successfully."
    # Optionally touch a file to indicate success
    touch "/tmp/$1-$2-client-done"
  else
    echo "Client encountered an error."
    # Optionally create a different indicator for error
    touch "/tmp/$1-$2-client-error"
  fi

  echo -e "Computation done, check out the log file ${GREEN}$1-$2_client.log${NC}"
fi
