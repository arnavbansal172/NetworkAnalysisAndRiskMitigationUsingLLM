import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional

# --- Hugging Face Hub Interaction ---
try:
    from huggingface_hub import snapshot_download, HfFolder
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except ImportError:
    print("[ERROR] `huggingface_hub` library not found. Please install it: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Model Definitions ---
# Define models here for easy management
SUPPORTED_MODELS = {
    "tinyllama": {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "default_dir": "tinyllama_1.1b_chat" # Subdirectory name
    },
    "openllama3b": {
        "repo_id": "openlm-research/open_llama_3b_v2",
        "default_dir": "open_llama_3b_v2"
    },
    # Add other models here if needed in the future
    # "example_model": {
    #     "repo_id": "user/example-model-name",
    #     "default_dir": "example_model"
    # }
}

DEFAULT_MODEL_KEY = "tinyllama" # Default to the primary model for this project

# --- Main Download Function ---
def download_hf_model(
    model_key: str,
    save_directory: Optional[Path] = None,
    hf_token: Optional[str] = None,
    use_auth_token: bool = True
):
    """
    Downloads a model snapshot from Hugging Face Hub.

    Args:
        model_key: The key corresponding to the model in SUPPORTED_MODELS.
        save_directory: The specific directory to save the model. If None,
                        a default path ('models/base/<model_default_dir>') is used.
        hf_token: Hugging Face Hub token (overrides cached token if provided).
        use_auth_token: Whether to try using the cached Hugging Face token.

    Returns:
        The Path object to the download directory if successful, None otherwise.
    """
    if model_key not in SUPPORTED_MODELS:
        logger.error(f"Model key '{model_key}' not found in supported models: {list(SUPPORTED_MODELS.keys())}")
        return None

    model_info = SUPPORTED_MODELS[model_key]
    repo_id = model_info["repo_id"]

    # Determine save path
    if save_directory is None:
        # Construct default path: models/base/<model_default_dir>
        save_path = Path("models") / "base" / model_info["default_dir"]
    else:
        save_path = Path(save_directory) # Ensure it's a Path object

    logger.info(f"Attempting to download '{repo_id}' ({model_key}) to: {save_path}")

    # Ensure parent directory exists
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured parent directory exists: {save_path.parent}")
    except OSError as e:
        logger.error(f"Could not create parent directory {save_path.parent}: {e}")
        return None

    # Determine authentication token
    token_to_use = hf_token
    if token_to_use is None and use_auth_token:
        # Try to get token from cache if not explicitly provided
        token_to_use = HfFolder.get_token()
        if token_to_use:
            logger.info("Using cached Hugging Face token for download.")
        else:
             logger.info("No explicit or cached Hugging Face token found. Downloading public model.")

    # Download the snapshot
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
            local_dir_use_symlinks=False, # Recommended for better cross-platform compatibility
            token=token_to_use,
            # ignore_patterns=["*.safetensors"], # Example: exclude specific file types if needed
            # resume_download=True, # Resume interrupted downloads
        )
        logger.info(f"Model '{repo_id}' downloaded successfully to {save_path}")
        return save_path
    except RepositoryNotFoundError:
         logger.error(f"Repository not found on Hugging Face Hub: '{repo_id}'. Check the model ID.")
         return None
    except HfHubHTTPError as e:
        logger.error(f"HTTP error during download: {e}")
        if e.response.status_code == 401:
             logger.error("Authentication error (401). Please ensure you are logged in (`huggingface-cli login`) or provide a valid token via --token.")
        elif e.response.status_code == 403:
             logger.error("Authorization error (403). You may not have access to this repository.")
        else:
             logger.error(f"Received status code {e.response.status_code}. Check network connection and model ID.")
        return None
    except Exception as e:
        # Catch other potential errors (network issues, disk space etc.)
        logger.error(f"An unexpected error occurred during download: {e}", exc_info=True)
        return None


# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download specified LLM models from Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help message
    )
    parser.add_argument(
        "--model", "-m",
        dest="model_key", # Store value in 'model_key'
        type=str,
        default=DEFAULT_MODEL_KEY,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Key of the model to download."
    )
    parser.add_argument(
        "--save_dir", "-s",
        dest="save_directory", # Store value in 'save_directory'
        type=Path,
        default=None,
        help="Specific directory to save the model files. If omitted, defaults to 'models/base/<model_dir_name>'."
    )
    parser.add_argument(
         "--token", "-t",
         dest="hf_token", # Store value in 'hf_token'
         type=str,
         default=None,
         help="Your Hugging Face Hub access token (optional, overrides cached token)."
    )
    parser.add_argument(
         "--no-auth",
         dest="use_auth_token", # Store False if flag is present
         action="store_false",
         help="Do not attempt to use the cached Hugging Face authentication token."
    )

    args = parser.parse_args()

    # Call the download function
    download_path = download_hf_model(
        model_key=args.model_key,
        save_directory=args.save_directory,
        hf_token=args.hf_token,
        use_auth_token=args.use_auth_token
    )

    if download_path:
        print(f"[SUCCESS] Model '{args.model_key}' is available at: {download_path.resolve()}")
        sys.exit(0)
    else:
        print(f"[FAILED] Could not download model '{args.model_key}'. Check logs for details.")
        sys.exit(1)