export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
export FLAGS_selected_xpus=1
export XPUSIM_DEVICE_MODEL=KUNLUN1
export XPU_PADDLE_TRAIN_L3_SIZE=13631488
export XPU_PADDLE_MAIN_STREAM=0

python -u run_classifier.py --task_name 'XNLI' \
                   --use_cuda false \
                   --use_xpu true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 16 \
                   --in_tokens false \
                   --init_pretraining_params chinese_L-12_H-768_A-12/params \
                   --data_dir data/XNLI-MT-1.0 \
                   --vocab_path chinese_L-12_H-768_A-12/vocab.txt \
                   --checkpoints save \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.1 \
                   --validation_steps 100 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --bert_config_path chinese_L-12_H-768_A-12/bert_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 10 \
                   --verbose true
