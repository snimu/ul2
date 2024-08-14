python main.py -c --logfile results_eight1.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 1.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1
python main.py -c --logfile results_eight2.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 1.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1 --ul2 --causal_divider 1000.0 --s_divider 1000.0 --r_divider 1.0 --x_divider 1.0  --causal_denoisers --alternate_denoisers --randomize_denoiser_settings --randomize_mask_width --no_special_tokens

python main.py -c --logfile results_eight3.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 5.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1
python main.py -c --logfile results_eight4.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 5.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1 --ul2 --causal_divider 1000.0 --s_divider 1000.0 --r_divider 1.0 --x_divider 1.0  --causal_denoisers --alternate_denoisers --randomize_denoiser_settings --randomize_mask_width --no_special_tokens

python main.py -c --logfile results_eight5.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 20.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1
python main.py -c --logfile results_eight6.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 20.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1 --ul2 --causal_divider 1000.0 --s_divider 1000.0 --r_divider 1.0 --x_divider 1.0  --causal_denoisers --alternate_denoisers --randomize_denoiser_settings --randomize_mask_width --no_special_tokens

python main.py -c --logfile results_eight7.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 40.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1
python main.py -c --logfile results_eight8.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 40.0 --seed 1600 --gpu_capacity_scalar 1.9 --num_runs 1 --max_epochs 1 --dataset fineweb --save_net --num_heads 1 --ul2 --causal_divider 1000.0 --s_divider 1000.0 --r_divider 1.0 --x_divider 1.0  --causal_denoisers --alternate_denoisers --randomize_denoiser_settings --randomize_mask_width --no_special_tokens

python main.py -c --logfile results_eight9.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 1.0 --seed 1700 --gpu_capacity_scalar 1.9 --num_runs 5 --max_epochs 4 --dataset fineweb --save_net --num_heads 1
python main.py -c --logfile results_eight10.csv -w --wandb_project ul2.eight --eval_every 50 --model_scale 1.0 --seed 1700 --gpu_capacity_scalar 1.9 --num_runs 5 --max_epochs 4 --dataset fineweb --save_net --num_heads 1 --ul2 --causal_divider 1000.0 --s_divider 1000.0 --r_divider 1.0 --x_divider 1.0  --causal_denoisers --alternate_denoisers --randomize_denoiser_settings --randomize_mask_width --no_special_tokens
