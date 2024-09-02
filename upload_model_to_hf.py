import argparse
from huggingface_hub import HfApi, login
import os
import shutil

def upload_model(storage_name, upload_name):
    # Initialize the Hugging Face API
    api = HfApi()

    # Login to Hugging Face (you'll need to set up your token as an environment variable)
    login()

    # Prepare the repository name
    repo_id = f"snimu/{upload_name}"

    # Get the directory and original filename
    dir_path = os.path.dirname(storage_name)
    temp_filename = "model.safetensors"

    try:
        # Rename the file to model.safetensors
        temp_path = os.path.join(dir_path, temp_filename)
        shutil.move(storage_name, temp_path)

        # Create the repository if it doesn't exist
        api.create_repo(repo_id, exist_ok=True)

        # Upload the model file
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=temp_filename,
            repo_id=repo_id,
            repo_type="model",
        )

        print(f"Model uploaded successfully to {repo_id}")

    finally:
        # Rename the file back to its original name
        if os.path.exists(temp_path):
            shutil.move(temp_path, storage_name)

def main():
    parser = argparse.ArgumentParser(description="Upload a PyTorch LLM .safetensors file to Hugging Face")
    parser.add_argument("--storage_name", required=True, help="Name of the file on disk to be uploaded")
    parser.add_argument("--upload_name", required=True, help="Name under which the model will be available on Hugging Face")

    args = parser.parse_args()

    upload_model(args.storage_name, args.upload_name)

if __name__ == "__main__":
    main()