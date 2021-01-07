export FLAGS_eager_delete_tensor_gb=1
export FLAGS_selected_xpus=0
export XPUSIM_DEVICE_MODEL=KUNLUN1
export XPU_PADDLE_TRAIN_L3_SIZE=13631488
export XPU_PADDLE_MAIN_STREAM=0


python -u predict_classifier.py --task_name "XNLI" \
      --use_cuda false\
       --use_xpu true\
       --batch_size 64 \
       --data_dir data/XNLI-MT-1.0 \
       --vocab_path chinese_L-12_H-768_A-12/vocab.txt \
       --init_checkpoint checkpoints/step_70000.pdparams \
       --do_lower_case true \
       --max_seq_len 128 \
       --bert_config_path chinese_L-12_H-768_A-12/bert_config.json \
       --do_predict true \
       --save_inference_model_path save/new
