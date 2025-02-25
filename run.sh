#!/bin/bash

# Arguments
MODEL_PATH=$1          
ADAPTER_PATH=$2        
INPUT_PATH=$3          
OUTPUT_PATH=$4         

# Command to execute the Python script with the specified arguments
python Inference.py \
    --base_model_path "$MODEL_PATH" \
    --peft_path "$ADAPTER_PATH" \
    --test_data_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_PATH"


# bash ./run.sh kyara_finetune_checkpoint adapter_checkpoint hw3/data/private_test.json prediction.json
