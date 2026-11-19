#!/usr/bin/env bash
# COVID-19: four independent prediction-length commands in one script.
# Run all: bash scripts/TQNet/covid-19.sh
# Run selected: bash scripts/TQNet/covid-19.sh 24 [36 48 60]
set -euo pipefail
cd "$(dirname "$0")/../.."

model_name=FTQNet
root_path_name=./dataset/
data_path_name=Covid-19.csv
model_id_name=Covid-19
data_name=Covid-19
seq_len=36

# Select which independent blocks to execute.
run_24=0
run_36=0
run_48=0
run_60=0
if [[ $# -eq 0 ]]; then
    run_24=1
    run_36=1
    run_48=1
    run_60=1
else
    for requested_len in "$@"; do
        case "$requested_len" in
            24) run_24=1 ;;
            36) run_36=1 ;;
            48) run_48=1 ;;
            60) run_60=1 ;;
            *) echo "Unsupported COVID-19 horizon: $requested_len" >&2; exit 2 ;;
        esac
    done
fi

# ==========================================================
# COVID-19 | H=24
# ==========================================================
if [[ $run_24 -eq 1 ]]; then
    pred_len=24
    for random_seed in 2021
    do
        python -u run.py \
          --is_training 1 \
          --root_path "$root_path_name" \
          --data_path "$data_path_name" \
          --model_id "${model_id_name}_${seq_len}_${pred_len}" \
          --model "$model_name" \
          --data "$data_name" \
          --features M \
          --target OT \
          --freq d \
          --seq_len "$seq_len" \
          --pred_len "$pred_len" \
          --label_len 0 \
          --enc_in 948 \
          --cycle 4 \
          --d_model 512 \
          --dropout 0.10 \
          --flinear_alpha 0.9 \
          --flinear_c 1.0 \
          --flinear_groups 8 \
          --train_epochs 100 \
          --patience 5 \
          --batch_size 8 \
          --learning_rate 0.0005 \
          --loss mae \
          --lradj type1 \
          --num_workers 1 \
          --itr 1 \
          --random_seed "$random_seed" \
          --covid_calibrated
    done
fi

# ==========================================================
# COVID-19 | H=36
# ==========================================================
if [[ $run_36 -eq 1 ]]; then
    pred_len=36
    for random_seed in 2021
    do
        python -u run.py \
          --is_training 1 \
          --root_path "$root_path_name" \
          --data_path "$data_path_name" \
          --model_id "${model_id_name}_${seq_len}_${pred_len}" \
          --model "$model_name" \
          --data "$data_name" \
          --features M \
          --target OT \
          --freq d \
          --seq_len "$seq_len" \
          --pred_len "$pred_len" \
          --label_len 0 \
          --enc_in 948 \
          --cycle 4 \
          --d_model 512 \
          --dropout 0.10 \
          --flinear_alpha 0.9 \
          --flinear_c 1.0 \
          --flinear_groups 8 \
          --train_epochs 100 \
          --patience 5 \
          --batch_size 8 \
          --learning_rate 0.0005 \
          --loss mae \
          --lradj type1 \
          --num_workers 1 \
          --itr 1 \
          --random_seed "$random_seed" \
          --covid_calibrated
    done
fi

# ==========================================================
# COVID-19 | H=48
# ==========================================================
if [[ $run_48 -eq 1 ]]; then
    pred_len=48
    for random_seed in 2021
    do
        python -u run.py \
          --is_training 1 \
          --root_path "$root_path_name" \
          --data_path "$data_path_name" \
          --model_id "${model_id_name}_${seq_len}_${pred_len}" \
          --model "$model_name" \
          --data "$data_name" \
          --features M \
          --target OT \
          --freq d \
          --seq_len "$seq_len" \
          --pred_len "$pred_len" \
          --label_len 0 \
          --enc_in 948 \
          --cycle 4 \
          --d_model 512 \
          --dropout 0.00 \
          --flinear_alpha 0.9 \
          --flinear_c 1.0 \
          --flinear_groups 8 \
          --train_epochs 100 \
          --patience 5 \
          --batch_size 8 \
          --learning_rate 0.01 \
          --loss mae \
          --lradj type1 \
          --num_workers 1 \
          --itr 1 \
          --random_seed "$random_seed" \
          --covid_calibrated
    done
fi

# ==========================================================
# COVID-19 | H=60
# ==========================================================
if [[ $run_60 -eq 1 ]]; then
    pred_len=60
    for random_seed in 2021
    do
        python -u run.py \
          --is_training 1 \
          --root_path "$root_path_name" \
          --data_path "$data_path_name" \
          --model_id "${model_id_name}_${seq_len}_${pred_len}" \
          --model "$model_name" \
          --data "$data_name" \
          --features M \
          --target OT \
          --freq d \
          --seq_len "$seq_len" \
          --pred_len "$pred_len" \
          --label_len 0 \
          --enc_in 948 \
          --cycle 4 \
          --d_model 512 \
          --dropout 0.10 \
          --flinear_alpha 0.95 \
          --flinear_c 1.0 \
          --flinear_groups 8 \
          --train_epochs 100 \
          --patience 5 \
          --batch_size 8 \
          --learning_rate 0.01 \
          --loss mae \
          --lradj type1 \
          --num_workers 1 \
          --itr 1 \
          --random_seed "$random_seed" \
          --covid_calibrated
    done
fi
