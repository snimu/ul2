torchrun --nproc_per_node=8 train_gpt2_muon.py --input-bin="../edu_fineweb100B/edu_fineweb_train*" --input-val-bin="../edu_fineweb100B/edu_fineweb_val*" --wandb-project="ul2.gpt2.muon" --save-every=10000 --hf-repo="" --use-mask
torchrun --nproc_per_node=8 train_gpt2_muon.py --input-bin="../edu_fineweb100B/edu_fineweb_train*" --input-val-bin="../edu_fineweb100B/edu_fineweb_val*" --wandb-project="ul2.gpt2.muon" --save-every=10000 --hf-repo=""

torchrun --nproc_per_node=8 train_gpt2_muon.py --wandb-project="ul2.gpt2.muon.large" --n-layer=64 --n-embd=1792 --n-head=14 --device-batch-size=6 --warmdown-iters=20000 --seed=1234 --val-loss-every=650 --save-every=50000 --hf-repo="" --use-mask
torchrun --nproc_per_node=8 train_gpt2_muon.py --wandb-project="ul2.gpt2.muon.large" --n-layer=64 --n-embd=1792 --n-head=14 --device-batch-size=6 --warmdown-iters=20000 --seed=1234 --val-loss-every=650 --save-every=50000 --hf-repo=""

torchrun --nproc_per_node=8 train_gpt2_muon.py --wandb-project="ul2.gpt2.muon.large" --n-layer=64 --n-embd=1792 --n-head=14 --device-batch-size=6 --warmdown-iters=20000 --seed=1234 --val-loss-every=650 --save-every=50000 --hf-repo="" --learning-rate 0.0014 --warmup-iters 10 --from-step=100000 --use-mask
torchrun --nproc_per_node=8 train_gpt2_muon.py --wandb-project="ul2.gpt2.muon.large" --n-layer=64 --n-embd=1792 --n-head=14 --device-batch-size=6 --warmdown-iters=20000 --seed=1234 --val-loss-every=650 --save-every=50000 --hf-repo="" --learning-rate 0.0014 --warmup-iters 10 --from-step=100000
