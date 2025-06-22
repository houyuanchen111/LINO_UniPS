import os
from huggingface_hub import snapshot_download



def cache_weights(weights_dir: str) -> dict:
    os.makedirs(weights_dir, exist_ok=True)
    model_ids = [
        "Stable-X/trellis-normal-v0-1",
        "houyuanchen/lino"
    ]
    cached_paths = {}
    for model_id in model_ids:
        print(f"Caching weights for: {model_id}")
        # Check if the model is already cached
        local_path = os.path.join(weights_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            print(f"Already cached at: {local_path}")
            cached_paths[model_id] = local_path
            continue
        # Download the model and cache it
        print(f"Downloading and caching model: {model_id}")
        # Use snapshot_download to download the model
        local_path = snapshot_download(repo_id=model_id, local_dir=os.path.join(weights_dir, model_id.split("/")[-1]), force_download=False)
        cached_paths[model_id] = local_path
        print(f"Cached at: {local_path}")

    return cached_paths

if __name__ == "__main__":
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    cache_weights(weights_dir)