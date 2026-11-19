#!/usr/bin/env bash
# Sliding training windows; each H has one fixed tail-H test origin and is trained independently.
# Shared FTQNet is unchanged: 4 heads, attention dropout=0.5 for all horizons.
# Boolean action flags use their existing False defaults: AMP, multi-GPU, FLOP-only.
# Run all horizons without arguments, or select horizons: bash influenza_us.sh 4 8
set -euo pipefail
cd "$(dirname "$0")/../.."
run_4=0; run_8=0; run_12=0
if [[ $# -eq 0 ]]; then
    run_4=1; run_8=1; run_12=1
else
    for horizon in "$@"; do
        case "$horizon" in
            4) run_4=1 ;;
            8) run_8=1 ;;
            12) run_12=1 ;;
            *) echo "Only 4, 8 and 12 weeks are supported" >&2; exit 2 ;;
        esac
    done
fi

# ===========================================================
# Influenza-US | 4 weeks | independent training
# ===========================================================
if [[ $run_4 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "Influenza-US_24_4_fixed" \
      --model FTQNet --des fixed \
      --data "Influenza-US" --root_path ./dataset/ --data_path "Influenza US.csv" \
      --features M --target ILITOTAL --freq w --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 24 --label_len 0 --pred_len 4 \
      --enc_in 53 --cycle 52 --use_revin 1 \
      --d_model 64 --dropout 0.1 --model_type mlp \
      --flinear_alpha 0.5 --flinear_c 0.0001 --flinear_groups 2 \
      --train_epochs 50 --patience 10 \
      --batch_size 128 --learning_rate 0.001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 42 \
      --use_gpu True --gpu 0 --devices 0
fi

# # ===========================================================
# # Influenza-US | 8 weeks | independent training
# # ===========================================================
if [[ $run_8 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "Influenza-US_24_8_fixed" \
      --model FTQNet --des fixed \
      --data "Influenza-US" --root_path ./dataset/ --data_path "Influenza US.csv" \
      --features M --target ILITOTAL --freq w --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 24 --label_len 0 --pred_len 8 \
      --enc_in 53 --cycle 52 --use_revin 1 \
      --d_model 64 --dropout 0.1 --model_type mlp \
      --flinear_alpha 0.9 --flinear_c 0.01 --flinear_groups 4 \
      --train_epochs 50 --patience 10 \
      --batch_size 128 --learning_rate 0.00001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 42 \
      --use_gpu True --gpu 0 --devices 0
fi

# # ===========================================================
# # Influenza-US | 12 weeks | independent training
# # ===========================================================
if [[ $run_12 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "Influenza-US_24_12_fixed" \
      --model FTQNet --des fixed \
      --data "Influenza-US" --root_path ./dataset/ --data_path "Influenza US.csv" \
      --features M --target ILITOTAL --freq w --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 24 --label_len 0 --pred_len 12 \
      --enc_in 53 --cycle 52 --use_revin 1 \
      --d_model 64 --dropout 0.1 --model_type mlp \
      --flinear_alpha 0.9 --flinear_c 1.0 --flinear_groups 4 \
      --train_epochs 50 --patience 10 \
      --batch_size 128 --learning_rate 0.001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 42 \
      --use_gpu True --gpu 0 --devices 0
fi
