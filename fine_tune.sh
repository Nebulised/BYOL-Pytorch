MODEL_PARAM_FILE="parameters/model_params.yaml"
FINE_TUNE_PARAM_FILE="parameters/fine_tune_params.yaml"

OUTPUT_FOLDER_PATH=""
PATH_TO_DATASET=""
MODEL_PATH=""

for lr in 0.1 0.03162278 0.01 0.00316228 0.001 0.00031623 0.0001; do
  bash update_yaml_param.sh "$FINE_TUNE_PARAM_FILE" "optimiser_params-lr" "$lr"
  for weight_decay in 0.000010 0.000022 0.000046 0.000100 0.000215 0.000464 0.001000 0; do;
    bash update_yaml_param.sh "$FINE_TUNE_PARAM_FILE" "optimiser_params-weight_decay" "$((weight_decay/lr))"
    python main.py --model-output-folder-path "$OUTPUT_FOLDER_PATH" --dataset-type "cifar10" --dataset-path "$PATH_TO_DATASET" --run-type "fine-tune" --gpu 0 --model-path "$MODEL_PATH" --num-workers 6 --run-para-file-path "$FINE_TUNE_PARAM_FILE" --model-param-file-path "$MODEL_PARAM_FILE"

  done

done