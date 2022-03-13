DATA_DIR='/data/private/yzh/projs/Robust-Prefix-Tuning/data'
MODEL_ROOT_DIR='/data/private/yzh/projs/Robust-Prefix-Tuning/saved_models'
python generate_adv_insent.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"
