# download_llama_3b.py
from huggingface_hub import snapshot_download
import argparse
import os

MODEL_ID = "openlm-research/open_llama_3b_v2"
DEFAULT_SAVE_DIR = "models/base/open_llama_3b_v2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Download the {MODEL_ID} model from Hugging Face Hub.")
    parser.add_argument(
        "--save_directory",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory to save the model files. Default: {DEFAULT_SAVE_DIR}"
    )
    parser.add_argument(
         "--token",
         type=str,
         default=None,
         help="Your Hugging Face Hub access token (if needed for private models, though OpenLlama should be public)."
    )

    args = parser.parse_args()

    print(f"Attempting to download {MODEL_ID} to {args.save_directory}...")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=args.save_directory,
            local_dir_use_symlinks=False, # Recommended for Windows compatibility and simplicity
            token=args.token,
            # ignore_patterns=["*.safetensors"], # Example: if you only want pytorch_model.bin
        )
        print(f"Model downloaded successfully to {args.save_directory}")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("Please ensure you have internet connectivity and sufficient disk space.")
        if "401 Client Error" in str(e):
             print("You might need to log in using `huggingface-cli login` or provide a token via the --token argument.")

# How to run: python download_llama_3b.py --save_directory path/to/save