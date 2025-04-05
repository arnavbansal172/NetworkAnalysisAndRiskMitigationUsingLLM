import logging
import argparse
import os
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    logging as hf_logging,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import load_dataset
from trl import SFTTrainer

# Use the utils module from the same directory
from .utils import format_prompt_chatml

# --- Logging Setup ---
# Reduce verbosity from Hugging Face libraries
hf_logging.set_verbosity_warning()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Constants ---
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEED = 42

# --- Main Training Function ---
def train(args):
    """Fine-tunes the TinyLlama-1.1B-Chat model."""
    set_seed(SEED)
    logger.info(f"Set random seed to {SEED}")

    logger.info("Starting fine-tuning process...")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Output Directory: {args.output_dir}")

    # --- 1. Load Dataset ---
    logger.info("Loading dataset...")
    try:
        dataset = load_dataset("json", data_files=str(args.data_path), split="train")
        logger.info(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
        if not dataset:
            logger.error("Dataset is empty. Aborting training.")
            return
        if not all(col in dataset.column_names for col in ["instruction", "input", "output"]):
             logger.error("Dataset is missing required columns: 'instruction', 'input', 'output'. Cannot format prompt.")
             return
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.data_path}: {e}", exc_info=True)
        return

    # --- 2. Load Tokenizer ---
    logger.info(f"Loading tokenizer for {args.base_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        # TinyLlama Chat should have pad token = eos token if loaded correctly
        if tokenizer.pad_token_id is None:
            logger.warning("Tokenizer missing pad token; setting pad_token_id to eos_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Ensure padding side is right for Causal LM
        tokenizer.padding_side = 'right'
        logger.info(f"Tokenizer loaded. Pad token ID: {tokenizer.pad_token_id}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        return

    # --- 3. Configure Quantization (QLoRA) ---
    use_qlora = args.use_qlora
    if use_qlora:
        logger.info("Using QLoRA (4-bit quantization).")
        try:
             compute_dtype = getattr(torch, args.bnb_compute_dtype)
        except AttributeError:
             logger.warning(f"Compute dtype {args.bnb_compute_dtype} not found in torch. Defaulting to float16.")
             compute_dtype = torch.float16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_use_double_quant,
        )
        logger.info(f"QLoRA Config: Quant Type={args.bnb_4bit_quant_type}, Compute DType={compute_dtype}, Double Quant={args.bnb_use_double_quant}")
    else:
        logger.info("Not using QLoRA.")
        bnb_config = None

    # --- 4. Load Base Model ---
    logger.info(f"Loading base model {args.base_model}...")
    # Determine torch_dtype if not using QLoRA
    model_dtype = None
    if not use_qlora:
        if args.bf16 and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            logger.info("Using bfloat16 for base model.")
        elif args.fp16:
            model_dtype = torch.float16
            logger.info("Using float16 for base model.")
        else:
            logger.info("Using default torch dtype (likely float32) for base model.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config, # Applied only if use_qlora is True
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=model_dtype, # Applied only if not use_qlora
            # use_flash_attention_2=True, # Optional: May speed up training if installed & supported
        )
        # Set model pad token id consistent with tokenizer
        model.config.pad_token_id = tokenizer.pad_token_id

        logger.info("Base model loaded.")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}", exc_info=True)
        # Provide more specific advice for common errors like OOM
        if "out of memory" in str(e).lower():
             logger.error("CUDA Out of Memory: Try using --use_qlora, reducing --batch_size, or enabling --gradient_checkpointing.")
        return

    # --- 5. Configure PEFT (LoRA) ---
    if use_qlora:
        logger.info("Preparing model for k-bit training...")
        # Gradient checkpointing is recommended for QLoRA
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        logger.info(f"Gradient Checkpointing active: {args.gradient_checkpointing}")
    elif args.gradient_checkpointing:
        # Enable gradient checkpointing without QLoRA if requested
        model.gradient_checkpointing_enable()
        logger.info(f"Gradient Checkpointing active: {args.gradient_checkpointing}")


    logger.info("Configuring LoRA...")
    # Common target modules for Llama-like architectures including TinyLlama
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA adapter
    try:
        # Check if model is already PEFT model (e.g. from resuming)
        if not isinstance(model, PeftModel):
            model = get_peft_model(model, lora_config)
            logger.info("LoRA adapter applied.")
            model.print_trainable_parameters()
        else:
            logger.info("Model is already a PeftModel, likely resuming.")
    except ValueError as ve:
         logger.error(f"Error applying LoRA. Target modules might be incorrect for {args.base_model}. Error: {ve}")
         logger.info(f"Available module names: {list(dict(model.named_modules()).keys())}")
         return
    except Exception as e:
         logger.error(f"Unexpected error applying LoRA: {e}", exc_info=True)
         return


    # --- 6. Configure Training Arguments ---
    logger.info("Configuring training arguments...")
    # Determine fp16/bf16 flags based on QLoRA status and args
    fp16_flag = not use_qlora and args.fp16
    bf16_flag = not use_qlora and args.bf16 and torch.cuda.is_bf16_supported()
    if use_qlora and (args.fp16 or args.bf16):
         logger.warning("fp16/bf16 flags have no effect when --use_qlora is enabled.")

    training_arguments = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps, # If > 0, overrides epochs
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optim="paged_adamw_32bit" if use_qlora else "adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=fp16_flag,
        bf16=bf16_flag,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_limit,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="tensorboard", # Or "wandb" if configured
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=SEED,
        # evaluation_strategy="steps", # Uncomment if using eval dataset
        # eval_steps=args.eval_steps,   # Uncomment if using eval dataset
        # dataloader_num_workers=4, # Optional: May speed up data loading
    )
    logger.info("Training arguments configured.")

    # --- 7. Initialize Trainer (SFTTrainer) ---
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset, # Or dataset["train"] if split
        # eval_dataset=eval_dataset, # Pass eval dataset if split
        peft_config=lora_config, # Pass LoRA config if using PEFT
        tokenizer=tokenizer,
        args=training_arguments,
        formatting_func=format_prompt_chatml, # Use our chatML formatter
        max_seq_length=args.max_seq_length,
        # packing=True, # Optional: set packing=True for potential speedup with short sequences
    )
    logger.info("SFTTrainer initialized.")

    # --- 8. Start Training ---
    logger.info("Starting model training...")
    try:
        resume_from_checkpoint = args.resume_from if args.resume_from else None
        logger.info(f"Resume from checkpoint: {resume_from_checkpoint or False}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training completed successfully.")

        # --- 9. Save Model & Tokenizer ---
        logger.info(f"Saving final fine-tuned model adapter to {args.output_dir}...")
        # Save adapter, tokenizer, and training args
        trainer.save_model(str(args.output_dir)) # Saves adapter config & weights
        tokenizer.save_pretrained(str(args.output_dir))
        # Good practice to save training args too
        torch.save(args, args.output_dir / "training_args.bin")

        # Log metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() # Saves optimizer state etc. for resuming
        logger.info("Model adapter, tokenizer, metrics, and training state saved.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama-1.1B-Chat for Network Vulnerability Assessment")

    # Paths
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the training data (JSONL file).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the fine-tuned adapter.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL_NAME, help="Base model ID from Hugging Face.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume training.")

    # Training Strategy
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if > 0).")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device training batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm for clipping.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True, help="Enable gradient checkpointing.")

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps.")
    parser.add_argument("--save_limit", type=int, default=2, help="Max number of checkpoints to keep.")

    # QLoRA Arguments
    parser.add_argument("--use_qlora", action='store_true', default=True, help="Use QLoRA (4-bit quantization). Set to false to disable.")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="QLoRA quant type ('nf4', 'fp4').")
    parser.add_argument("--bnb_compute_dtype", type=str, default="bfloat16", help="QLoRA compute dtype ('bfloat16', 'float16').")
    parser.add_argument("--bnb_use_double_quant", action='store_true', default=False, help="Use double quantization for QLoRA.")

    # LoRA Arguments
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.") # Slightly lower dropout maybe

    # Mixed Precision (if not using QLoRA)
    parser.add_argument("--fp16", action='store_true', default=False, help="Enable fp16 training (ignored if using QLoRA).")
    parser.add_argument("--bf16", action='store_true', default=False, help="Enable bf16 training (ignored if using QLoRA).")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train(args)
