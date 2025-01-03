# /// script
# requires-python = "==3.12"
# dependencies = [
#   "huggingface-hub",
# ]
# ///

"""By Claude"""

from pathlib import Path

from huggingface_hub import HfApi, create_repo

def upload_to_huggingface(local_dir: str, repo_name: str, token: str):
    """
    Upload binary shards to HuggingFace
    
    Args:
        local_dir: Directory containing the .bin files
        repo_name: Name of the HF repo (e.g. 'snimu/finemath-4plus-tiktokenized')
        token: HuggingFace API token
    """
    # Initialize the HF API
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=False,
            token=token
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e
    
    # Create dataset card
    readme_content = f"""---
license: apache-2.0
---

# {repo_name}

This dataset contains GPT-2 tokenized shards of the FineWeb-4plus dataset.
Each shard is stored in a binary format with the following structure:
- First comes a header with 256 int32s
- The tokens follow, each as uint16 (GPT-2 format)

The first shard is the validation set, subsequent shards are training data.

Original dataset: HuggingFaceTB/finemath
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    # Upload README
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
        token=token
    )
    
    # Upload all .bin files
    local_files = sorted(Path(local_dir).glob("*.bin"))
    for file_path in local_files:
        print(f"Uploading {file_path.name}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload tokenized shards to HuggingFace")
    parser.add_argument("--local_dir", type=str, required=True, help="Directory containing the .bin files")
    parser.add_argument("--repo_name", type=str, default="snimu/finemath-4plus-tiktokenized", 
                        help="Name of the HF repo")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
    
    args = parser.parse_args()
    upload_to_huggingface(args.local_dir, args.repo_name, args.token)
