#!/usr/bin/env bash
# Wearable: same run.py and original master result.txt/results/*.txt format.
set -euo pipefail
cd "$(dirname "$0")/../.."
model_name=FTQNet
root_path_name=./dataset/
data_path_name=wearable.csv
model_id_name=Wearable
data_name=Wearable
seq_len=576
if [[ $# -gt 0 ]]; then horizon_list=("$@"); else horizon_list=(12 48 144 288); fi
for pred_len in "${horizon_list[@]}"; do
  case $pred_len in 12|48|144|288) ;; *) echo "Unsupported Wearable horizon: $pred_len" >&2; exit 2 ;; esac
  for random_seed in 2021; do
    python -u run.py \
      --is_training 1 --model_id "${model_id_name}_${seq_len}_${pred_len}" \
      --model "$model_name" --data "$data_name" \
      --root_path "$root_path_name" --data_path "$data_path_name" \
      --features M --target HR --freq 5min \
      --seq_len "$seq_len" --pred_len "$pred_len" --label_len 0 \
      --enc_in 19 --cycle 4 --d_model 64 --dropout 0.242352 \
      --flinear_alpha 0.9 --flinear_c 1.0 --flinear_groups 8 \
      --train_epochs 30 --patience 5 --batch_size 128 --learning_rate 0.009135185 \
      --loss mse --lradj type1 --num_workers 0 --itr 1 --random_seed "$random_seed"
  done
done
