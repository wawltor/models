export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../../../
export FLAGS_call_stack_level=2
python run_pretrain_static_single.py --model_name_or_path gpt2-small-en --input_dir "./input_data/final_dataset"\
    --output_dir "output"\
    --learning_rate 0.00015\
    --weight_decay 0.01\
    --save_steps 2000\
    --max_steps 320000\
    --warmup_rate .01\
    --batch_size 32\
    --grad_clip 1.0\
    --logging_steps 1\
    --scale_loss 1024\
    --use_amp True\
