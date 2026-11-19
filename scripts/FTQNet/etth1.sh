model_name=FTQNet

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=96
for pred_len in 96 192 336 720
do
case $pred_len in
  96)
    flinear_alpha=0.85
    flinear_c=0.0001
    flinear_groups=16
    ;;
  192)
    flinear_alpha=0.4
    flinear_c=0.0001
    flinear_groups=32
    ;;
  336)
    flinear_alpha=0.8
    flinear_c=0.75
    flinear_groups=16
    ;;
  720)
    flinear_alpha=0.73
    flinear_c=1.1
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
      --enc_in 7 \
      --cycle 24 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed \
      --flinear_alpha $flinear_alpha \
      --flinear_c $flinear_c \
      --flinear_groups $flinear_groups
done
done

