#!/usr/bin/env bash
# MicroSTNet published author-code protocol for FTQNet (not independent test).
# Global Z-score +1; sequential 70/30; reuse ~10% validation windows in train.
# Adam 1e-3 / L1 loss / 200 full epochs; select best validation MAE.
# Only new microbiome adapter is affected; other datasets are unchanged.
# Dataset and result files keep their original names and locations.
# Script options: no arguments runs both; pass 1 or 8 to run one.
set -euo pipefail
cd "$(dirname "$0")/../.."
run_1=0; run_8=0
if [[ $# -eq 0 ]]; then
    run_1=1; run_8=1
else
    for horizon in "$@"; do
        case "$horizon" in
            1) run_1=1 ;;
            8) run_8=1 ;;
            *) echo "Only horizons 1 and 8 are supported" >&2; exit 2 ;;
        esac
    done
fi

# ===========================================================
# M3-stool | input 12 samples | predict 1 | separate model run
# ===========================================================
if [[ $run_1 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "M3-stool_12_1" \
      --model FTQNet --des microbiome \
      --data "M3-stool" --root_path ./dataset/ --data_path "M3_gut_abun.csv" \
      --features M --target OT --freq d --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 12 --label_len 0 --pred_len 1 \
      --enc_in 0 --cycle 1 --use_revin 1 \
      --d_model 64 --dropout 0.0 --model_type mlp \
      --flinear_alpha 0.9 --flinear_c 1.0 --flinear_groups 4 \
      --train_epochs 200 --patience 201 \
      --batch_size 8 --learning_rate 0.001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 7 \
      --use_gpu True --gpu 0 --devices 0
fi

# ===========================================================
# M3-stool | input 12 samples | predict 8 | separate model run
# ===========================================================
if [[ $run_8 -eq 1 ]]; then
    python -u run.py \
      --is_training 1 --model_id "M3-stool_12_8" \
      --model FTQNet --des microbiome \
      --data "M3-stool" --root_path ./dataset/ --data_path "M3_gut_abun.csv" \
      --features M --target OT --freq d --embed timeF \
      --checkpoints ./checkpoints/ \
      --seq_len 12 --label_len 0 --pred_len 8 \
      --enc_in 0 --cycle 1 --use_revin 1 \
      --d_model 64 --dropout 0.3 --model_type mlp \
      --flinear_alpha 0.9 --flinear_c 0.1 --flinear_groups 2 \
      --train_epochs 200 --patience 201 \
      --batch_size 8 --learning_rate 0.001 \
      --loss mae --lradj constant --pct_start 0.3 \
      --num_workers 0 --itr 1 --random_seed 7 \
      --use_gpu True --gpu 0 --devices 0
fi
