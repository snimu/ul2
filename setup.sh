git clone https://github.com/snimu/ul2.git
cd ul2
pip install uv
uv venv
source ./venv/bin/activate
uv pip install torch torchvision
uv pip install -r requirements.txt
bash download_fineweb_edu.sh
echo "Now run 'export HF_API_TOKEN= ...' to get your token"
