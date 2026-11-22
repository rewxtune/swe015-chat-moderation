import sys
from huggingface_hub import login, create_repo, HfApi

repo = sys.argv[1]
login()
create_repo(repo, exist_ok=True)
HfApi().upload_folder(folder_path="models", repo_id=repo)
print(f"Yuklendi: https://huggingface.co/{repo}")
