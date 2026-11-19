model_name=FTQNet

root_path_name=./dataset/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS


seq_len=96
for pred_len in 12 24 48 96
do
case $pred_len in
  12)
    flinear_alpha=0.998
    flinear_c=0.5
    flinear_groups=16
    ;;
  24)
    flinear_alpha=0.99999
    flinear_c=1.0
    flinear_groups=8
    ;;
  48)
    flinear_alpha=0.99999
    flinear_c=1.0
    flinear_groups=32
    ;;
  96)
    flinear_alpha=0.8
    flinear_c=1.0
    flinear_groups=32
    ;;
esac
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 358 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --flinear_alpha $flinear_alpha \
      --flinear_c $flinear_c \
      --flinear_groups $flinear_groups \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done

