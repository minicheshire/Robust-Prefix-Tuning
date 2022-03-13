DATA_DIR='/data/private/yzh/projs/Robust-Prefix-Tuning/data_adv_train'
MODEL_ROOT_DIR='/data/private/yzh/projs/Robust-Prefix-Tuning/saved_models_adv_train'
python train_adv.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"
