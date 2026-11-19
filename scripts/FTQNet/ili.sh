#!/usr/bin/env bash
# ILI: same run.py, model_id, and result.txt/results/*.txt style as other scripts.
set -euo pipefail
cd "$(dirname "$0")/../.."
model_name=FTQNet
root_path_name=./dataset/
data_path_name=ILI.csv
model_id_name=ILI
data_name=ILI
seq_len=104
# Optional horizons as arguments; otherwise the standard four.
if [[ $# -gt 0 ]]; then horizon_list=("$@"); else horizon_list=(24 36 48 60); fi
for pred_len in "${horizon_list[@]}"; do
  case $pred_len in
    24) d_model=512; flinear_alpha=0.99; dropout=0.10; learning_rate=0.001 ;;
    36) d_model=512; flinear_alpha=0.90; dropout=0.15; learning_rate=0.001 ;;
    48) d_model=1024; flinear_alpha=0.999; dropout=0.05; learning_rate=0.0005 ;;
    60) d_model=1024; flinear_alpha=0.999; dropout=0.00; learning_rate=0.001 ;;
    *) echo "Unsupported ILI horizon: $pred_len" >&2; exit 2 ;;
  esac
  for random_seed in 2021; do
    python -u run.py \
      --is_training 1 --model_id "${model_id_name}_${seq_len}_${pred_len}" \
      --model "$model_name" --data "$data_name" \
      --root_path "$root_path_name" --data_path "$data_path_name" \
      --features M --target OT --freq w \
      --seq_len "$seq_len" --pred_len "$pred_len" --label_len 0 \
      --enc_in 7 --cycle 4 --d_model "$d_model" --dropout "$dropout" \
      --flinear_alpha "$flinear_alpha" --flinear_c 1.0 --flinear_groups 8 \
      --train_epochs 100 --patience 5 --batch_size 8 --learning_rate "$learning_rate" \
      --loss mae --lradj type1 --num_workers 1 --itr 1 --random_seed "$random_seed"
  done
done
