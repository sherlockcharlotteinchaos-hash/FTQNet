#!/usr/bin/env bash
# Sliding training windows; each H has one fixed tail-H test origin and is trained independently.
# Shared FTQNet internals stay fixed: 4 attention heads, attention dropout=0.5.
# Boolean action flags use their existing False defaults: AMP, multi-GPU, FLOP-only.
# Run all horizons without arguments, or select horizons: bash chickenpox_hungary.sh 4 8
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
# Chickenpox-Hungary | 4 weeks | independent training
# ===========================================================
if [[ $run_4 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "Chickenpox-Hungary_24_4_fixed" \
      --model FTQNet --des fixed \
      --data "Chickenpox-Hungary" --root_path ./dataset/ --data_path "hungary_chickenpox.csv" \
      --features M --target "OT" --freq w --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 24 --label_len 0 --pred_len 4 \
      --enc_in 20 --cycle 52 --use_revin 0 \
      --d_model 128 --dropout 0.0 --model_type mlp \
      --flinear_alpha 0.9 --flinear_c 1.0 --flinear_groups 8 \
      --train_epochs 50 --patience 10 \
      --batch_size 32 --learning_rate 0.001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 42 \
      --use_gpu True --gpu 0 --devices 0
fi

# ===========================================================
# Chickenpox-Hungary | 8 weeks | independent training
# ===========================================================
if [[ $run_8 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "Chickenpox-Hungary_24_8_fixed" \
      --model FTQNet --des fixed \
      --data "Chickenpox-Hungary" --root_path ./dataset/ --data_path "hungary_chickenpox.csv" \
      --features M --target "OT" --freq w --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 24 --label_len 0 --pred_len 8 \
      --enc_in 20 --cycle 52 --use_revin 0 \
      --d_model 128 --dropout 0.0 --model_type mlp \
      --flinear_alpha 0.99 --flinear_c 1.0 --flinear_groups 8 \
      --train_epochs 50 --patience 10 \
      --batch_size 32 --learning_rate 0.001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 42 \
      --use_gpu True --gpu 0 --devices 0
fi

# ===========================================================
# Chickenpox-Hungary | 12 weeks | independent training
# ===========================================================
if [[ $run_12 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "Chickenpox-Hungary_24_12_fixed" \
      --model FTQNet --des fixed \
      --data "Chickenpox-Hungary" --root_path ./dataset/ --data_path "hungary_chickenpox.csv" \
      --features M --target "OT" --freq w --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 24 --label_len 0 --pred_len 12 \
      --enc_in 20 --cycle 52 --use_revin 0 \
      --d_model 128 --dropout 0.0 --model_type mlp \
      --flinear_alpha 0.99 --flinear_c 1.0 --flinear_groups 8 \
      --train_epochs 50 --patience 10 \
      --batch_size 32 --learning_rate 0.001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 42 \
      --use_gpu True --gpu 0 --devices 0
fi
