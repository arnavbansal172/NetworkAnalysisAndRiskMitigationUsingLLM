import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def load_data_from_jsonl(file_path: str | Path) -> List[Dict[str, Any]]:
    """Loads data efficiently from a JSON Lines file."""
    file_path = Path(file_path)
    data = []
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip(): # Ensure line is not empty
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON on line {line_num} in {file_path}: {line.strip()}")
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading data file {file_path}: {e}", exc_info=True)
        raise

def format_prompt_chatml(example: Dict[str, Any]) -> str:
    """
    Formats the prompt string using a chatML-like structure suitable for
    TinyLlama-1.1B-Chat models fine-tuned with this format.

    Input dictionary `example` must contain 'instruction', 'input', and 'output' keys.
    'output' should contain the target JSON string *including* braces.
    """
    instruction = example.get("instruction")
    input_context = example.get("input") # Packet features text
    output_json = example.get("output") # Target JSON string

    if not all([instruction, input_context, output_json]):
        logger.error(f"Missing required keys ('instruction', 'input', 'output') in example: {example}")
        # Return an empty string or raise an error to prevent malformed training data
        return ""

    # Construct the prompt using the specific chat format TinyLlama Chat expects
    # Ref: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0#chat-template
    system_prompt = "<|system|>\nYou are a network security analyst. Analyze the provided network packet features for potential vulnerabilities. Respond ONLY with a valid JSON object containing your analysis. Ensure the JSON is complete and correctly formatted.</s>"
    user_prompt = f"<|user|>\n{instruction}\n\nInput Network Data:\n{input_context}</s>"
    assistant_response = f"<|assistant|>\n{output_json}</s>" # Target JSON included here

    # Combine into a single string for SFTTrainer's 'text' field processing
    full_text = system_prompt + "\n" + user_prompt + "\n" + assistant_response

    return full_text