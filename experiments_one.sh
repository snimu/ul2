python main.py -cw --logfile results.csv --wandb_project ul2.one --model_scale 5.0 2.5 1.0 0.5 0.1 --seed 1000--gpu_capacity_scalar 0.8 
python main.py -cw --append --logfile results.csv --wandb_project ul2.one --model_scale 5.0 2.5 1.0 0.5 0.1 --seed 1000 --gpu_capacity_scalar 0.8 --ul2
python main.py -cw --append --logfile results.csv --wandb_project ul2.one --model_scale 5.0 2.5 1.0 0.5 0.1 --seed 1000 --gpu_capacity_scalar 0.8 --ul2 --causal_denoisers