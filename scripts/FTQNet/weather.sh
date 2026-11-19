model_name=FTQNet

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in 96 192 336 720
do
case $pred_len in
  96)
    flinear_alpha=0.9999
    flinear_c=0.9
    flinear_groups=32
    ;;
  192)
    flinear_alpha=0.9999
    flinear_c=1.0
    flinear_groups=8
    ;;
  336)
    flinear_alpha=0.99
    flinear_c=2.0
    flinear_groups=8
    ;;
  720)
    flinear_alpha=0.9
    flinear_c=1.1
    flinear_groups=8
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
      --enc_in 21 \
      --cycle 144 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --flinear_alpha $flinear_alpha \
      --flinear_c $flinear_c \
      --flinear_groups $flinear_groups \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done

