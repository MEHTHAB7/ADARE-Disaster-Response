import os
from huggingface_hub import HfApi

print("Starting direct API upload to Hugging Face...")
api = HfApi()

try:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Please set the HF_TOKEN environment variable.")
    else:
        api.upload_folder(
            folder_path=".",
            repo_id="Mehthab07/openenv-disaster-response",
            repo_type="space",
            token=token,
            ignore_patterns=[".git/*", "ui/node_modules/*", "upload_to_hf.py"]
        )
        print("Upload completed successfully!")
except Exception as e:
    print(f"Failed to upload: {e}")
