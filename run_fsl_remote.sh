#!/bin/bash

# Default arguments
CKPT_DIR="ssl/solo-learn/trained_models"
DATASETS_DIR="/workspace/datasets"
OUTPUT_FILE="fsl_results.txt"
DEVICE="cuda"
TARGET_METHOD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ckpt_dir)
      CKPT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    --datasets_dir)
      DATASETS_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    --device)
      DEVICE="$2"
      shift # past argument
      shift # past value
      ;;
    --method)
      TARGET_METHOD="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Running FSL evaluation..."
echo "Checkpoint Directory: $CKPT_DIR"
echo "Datasets Directory: $DATASETS_DIR"
echo "Output File: $OUTPUT_FILE"
echo "Device: $DEVICE"
if [ -n "$TARGET_METHOD" ]; then
    echo "Target Method: $TARGET_METHOD"
else
    echo "Target Method: ALL (simclr, byol, swav, vicreg)"
fi

# Create output file
touch $OUTPUT_FILE

# Determine methods to run
if [ -n "$TARGET_METHOD" ]; then
    METHODS="$TARGET_METHOD"
else
    METHODS="simclr byol swav vicreg"
fi

# Iterate over methods and seeds if applicable (adjust structure based on actual directory layout)
# Assuming structure: ssl/solo-learn/trained_models/<method>/<seed>/<checkpoint>

for method in $METHODS; do
    echo "Processing method: $method"
    
    echo "Evaluating method: $method"
    echo "----------------------------------------" >> $OUTPUT_FILE
    
    # Determine method checkpoint directory
    base_method_dir="$CKPT_DIR/$method"
    
    if [ ! -d "$base_method_dir" ]; then
        echo "Directory $base_method_dir not found. Skipping."
        continue
    fi
    
    # Dynamically find the seed directory (e.g. 0, 1, 2, 3)
    # We take the first subdirectory found (assuming one seed per method as per ls output)
    method_ckpt_dir=$(find "$base_method_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    
    if [ -z "$method_ckpt_dir" ]; then
        echo "No seed directory found in $base_method_dir, checking for ckpts in root..."
        method_ckpt_dir="$base_method_dir"
    fi
    
    echo "Checkpoint Dir: $method_ckpt_dir"
    
    # Run bulk evaluation
    # Output to separate JSON file for each method
    json_output="${method}_fsl_results.json"
    
    python3 fsl/evaluate_bulk.py \
        --ckpt_dir "$method_ckpt_dir" \
        --datasets_dir "$DATASETS_DIR" \
        --output_file "$json_output" \
        --every_n 5 \
        --n_way 5 \
        --n_shot 5 \
        --n_query 15 \
        --n_episodes 600 \
        --device "$DEVICE"
        
    echo "Done with $method. Saved to $json_output"
done

echo "Evaluation complete."

