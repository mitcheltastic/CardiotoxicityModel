import os
from dotenv import load_dotenv
from huggingface_hub import upload_file

load_dotenv()

upload_file(
    path_or_fileobj='hf_space/app.py',
    path_in_repo='app.py',
    repo_id='mitcheltastic/tdp-cipa-screening',
    repo_type='space',
    token=os.environ['HF_TOKEN'],
    commit_message='Add conformal prediction layer (alpha=0.154, q_hat=0.6275)',
)
print("done")
