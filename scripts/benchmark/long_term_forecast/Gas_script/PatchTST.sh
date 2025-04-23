export CUDA_VISIBLE_DEVICES=5

model_name=PatchTST

for pred_len in 96 192 336 720;do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Gas/ \
  --data_path 134312_data.csv \
  --model_id Exchange_96_${pred_len} \
  --model $model_name \
  --data Gas \
  --features M \
  --target value \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1
done
