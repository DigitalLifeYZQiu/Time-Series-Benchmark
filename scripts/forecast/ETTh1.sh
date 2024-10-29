export CUDA_VISIBLE_DEVICES=2
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
# ckpt_path=ckpt/Timer/Timer_forecast_1.0.ckpt
ckpt_path=ckpt/Timer/16B_timegpt_d1024_l8_p96_n64_new_full.ckpt
data=ETTh1

for subset_rand_ratio in 1
do
# train
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --task_name forecast \
  --is_training 0 \
  --is_finetuning 0 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/ETT-small/ \
  --data_path $data.csv \
  --data $data \
  --model_id etth1_sr_$subset_rand_ratio \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 2048 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu
done
