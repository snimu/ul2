python main.py -c --logfile results_eight1.csv --no_eval --model_scale 1.0 --seed 1600 --gpu_capacity_scalar 0.7 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 8 1
python main.py -c --logfile results_eight2.csv --no_eval --model_scale 1.0 --seed 1600 --gpu_capacity_scalar 0.7 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 8 1 --ul2 --causal_divider 1000.0 --s_divider 1000.0 --r_divider 1.0 --x_divider 1.0  --causal_denoisers --alternate_denoisers --randomize_denoiser_settings --randomize_mask_width --no_special_tokens

python main.py -c --logfile results_eight3.csv --no_eval --model_scale 5.0 --seed 1600 --gpu_capacity_scalar 1.8 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 8 1
python main.py -c --logfile results_eight4.csv --no_eval --model_scale 5.0 --seed 1600 --gpu_capacity_scalar 1.8 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 8 1 --ul2 --causal_divider 1000.0 --s_divider 1000.0 --r_divider 1.0 --x_divider 1.0  --causal_denoisers --alternate_denoisers --randomize_denoiser_settings --randomize_mask_width --no_special_tokens

