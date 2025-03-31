# src/llm_training/train.py
import logging
import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # For QLoRA
    TrainingArguments,
    # Trainer, # Use SFTTrainer for easier instruction tuning
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from trl import SFTTrainer # Using TRL library simplifies SFT

# Assuming utils.py is in the same directory or Python path is set correctly
from .utils import format_prompt # Use the formatting function

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Constants ---
DEFAULT_MODEL_NAME = "openlm-research/open_llama_3b_v2"

# --- Main Training Function ---
def train(args):
    """Fine-tunes the Open Llama 3B model."""

    logger.info("Starting fine-tuning process...")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Output Directory: {args.output_dir}")

    # --- 1. Load Dataset ---
    logger.info("Loading dataset...")
    try:
        # Load dataset directly using datasets library, assumes jsonl format
        dataset = load_dataset("json", data_files=args.data_path, split="train")
        # TODO: Add support for validation dataset if available
        logger.info(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
        if "instruction" not in dataset.column_names or \
           "input" not in dataset.column_names or \
           "output" not in dataset.column_names:
             logger.warning("Dataset might be missing 'instruction', 'input', or 'output' columns needed for formatting.")
             # If format_prompt handles missing keys gracefully, this might be okay, otherwise raise error.

    except Exception as e:
        logger.error(f"Failed to load dataset from {args.data_path}: {e}", exc_info=True)
        return

    # --- 2. Load Tokenizer ---
    logger.info(f"Loading tokenizer for {args.base_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        # Set padding token if it's not already set
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer missing pad token; using eos_token as pad token.")
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.padding_side = "right" # Optional: Often recommended for Causal LMs

        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        return

    # --- 3. Configure Quantization (QLoRA) ---
    # Improves memory efficiency, crucial for larger models on limited hardware
    use_qlora = args.use_qlora
    if use_qlora:
        logger.info("Using QLoRA (4-bit quantization).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if available
            bnb_4bit_use_double_quant=False, # Optional
        )
    else:
        logger.info("Not using QLoRA. Loading model in default precision (or float16 if specified).")
        bnb_config = None # No quantization

    # --- 4. Load Base Model ---
    logger.info(f"Loading base model {args.base_model}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config if use_qlora else None,
            device_map="auto", # Automatically distribute across GPUs if available
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not use_qlora and torch.cuda.is_bf16_supported() else torch.float16 # Use bf16/fp16 if not quantizing
        )
        # Resize token embeddings if pad token was added
        model.resize_token_embeddings(len(tokenizer))
        # Configure model pad token id
        model.config.pad_token_id = tokenizer.pad_token_id

        logger.info("Base model loaded.")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}", exc_info=True)
        return

    # --- 5. Configure PEFT (LoRA) ---
    # Parameter-Efficient Fine-Tuning reduces the number of trainable parameters
    if use_qlora:
        # Prepare model for k-bit training if using quantization
        model = prepare_model_for_kbit_training(model)

    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,                # Rank of the update matrices
        lora_alpha=args.lora_alpha,   # Alpha scaling factor
        target_modules=["q_proj", "v_proj"], # Common targets for Llama, adjust if needed
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # Apply LoRA adapter to the model
    model = get_peft_model(model, lora_config)
    logger.info("LoRA configured.")
    model.print_trainable_parameters() # Show how many parameters are being trained

    # --- 6. Configure Training Arguments ---
    logger.info("Configuring training arguments...")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optim="paged_adamw_32bit" if use_qlora else "adamw_torch", # Optimizer compatible with QLoRA
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        # Weight decay, warmup steps etc. can be added here
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_limit, # Limit number of checkpoints
        fp16=not use_qlora and torch.cuda.is_available(), # Enable mixed precision if not using QLoRA and GPU available
        bf16=not use_qlora and torch.cuda.is_bf16_supported(), # Use bfloat16 if supported (preferred over fp16)
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing, # Saves memory but slows down training
        # evaluation_strategy="steps", # Add if using eval dataset
        # eval_steps=args.eval_steps,    # Add if using eval dataset
        report_to="tensorboard", # Or "wandb", "none"
    )
    logger.info("Training arguments configured.")

    # --- 7. Initialize Trainer (SFTTrainer) ---
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        # eval_dataset=validation_dataset, # Add validation dataset here
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_arguments,
        # Use formatting_func to apply our prompt structure
        # SFTTrainer handles creating the 'text' column from dataset cols
        formatting_func=format_prompt,
        max_seq_length=args.max_seq_length, # Max sequence length for tokenization
        # packing=True, # Can potentially speed up training if sequences are short
    )
    logger.info("SFTTrainer initialized.")

    # --- 8. Start Training ---
    logger.info("Starting model training...")
    try:
        train_result = trainer.train()
        logger.info("Training completed successfully.")

        # --- 9. Save Model & Tokenizer ---
        logger.info(f"Saving fine-tuned model adapter to {args.output_dir}...")
        # Saves the LoRA adapter weights, not the full model
        trainer.save_model(args.output_dir)
        # Also save the tokenizer
        tokenizer.save_pretrained(args.output_dir)

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("Model adapter, tokenizer, and training state saved.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Open Llama 3B for Network Vulnerability Assessment")

    # Required arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data (JSONL file).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned adapter and tokenizer.")

    # Model arguments
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL_NAME, help="Base model name or path from Hugging Face.")
    parser.add_argument("--use_qlora", action='store_true', help="Use QLoRA (4-bit quantization) for training.")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Per device training batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--logging_steps", type=int, default=25, help="Log training metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument("--save_limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm for clipping.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler type (e.g., 'linear', 'cosine').")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory.")

    # LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)