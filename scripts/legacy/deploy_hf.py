import os
from huggingface_hub import upload_folder

# Automatically load HF_TOKEN from .env file if available
if 'HF_TOKEN' not in os.environ:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('HF_TOKEN='):
                    os.environ['HF_TOKEN'] = line.split('=', 1)[1].strip(' "\'')
                    break

token = os.environ.get('HF_TOKEN')
if not token:
    raise ValueError("HF_TOKEN environment variable not set.")

print("Uploading hf_space/ to Hugging Face...")
url = upload_folder(
    folder_path='hf_space',
    repo_id='mitcheltastic/tdp-cipa-screening',
    repo_type='space',
    token=token,
    commit_message='Switch to Random Forest with TreeSHAP and rank-11 conformal',
    ignore_patterns=['.venv/*', '__pycache__/*', '.venv', 'deployable_chantest_3class.joblib', 'shap_background.joblib']
)
print("Upload successful!")
print(f"Commit URL: {url}")
print("Space URL: https://huggingface.co/spaces/mitcheltastic/tdp-cipa-screening")
