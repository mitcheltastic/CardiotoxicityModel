import os
from huggingface_hub import upload_folder

token = os.environ.get('HF_TOKEN')
if not token:
    raise ValueError("HF_TOKEN environment variable not set.")

print("Uploading hf_space/ to Hugging Face...")
url = upload_folder(
    folder_path='hf_space',
    repo_id='mitcheltastic/tdp-cipa-screening',
    repo_type='space',
    token=token,
    commit_message='Add per-prediction SHAP /explain endpoint',
    ignore_patterns=['.venv/*', '__pycache__/*', '.venv']
)
print("Upload successful!")
print(f"Commit URL: {url}")
print("Space URL: https://huggingface.co/spaces/mitcheltastic/tdp-cipa-screening")
