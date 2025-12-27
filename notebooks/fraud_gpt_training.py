"""
Fine-tune language model on fraud patterns using LoRA.

Creates Fraud-GPT: domain-specific model for fraud summary generation.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType


class FraudGPTTrainer:
    """Fine-tune small LM with LoRA for fraud description generation."""

    def __init__(self, model_name="gpt2", data_dir="../data", output_dir="../models"):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup_lora_config(self):
        """Configure LoRA for parameter-efficient fine-tuning."""
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return config

    def prepare_training_data(self):
        """Ensure fraud_patterns.txt exists."""
        patterns_file = self.data_dir / "fraud_patterns.txt"

        if not patterns_file.exists():
            print(f"[WARNING] {patterns_file} not found. Generating sample patterns...")
            patterns = [
                "Rapid sequence of small transactions followed by large withdrawal",
                "Transaction from unusual geographic location within 2 hours",
                "Card used for category switch: fuel to jewelry in 30 minutes",
                "Multiple failed authentication attempts then successful login",
                "Account balance check followed by maximum withdrawal",
                "Device fingerprint mismatch with historical pattern",
                "Transaction during known offline hours for merchant",
                "Velocity spike: 10 transactions in 5 minutes",
                "Peer-to-peer transfer to new recipient then immediate cash-out",
                "Transaction amount exactly matches available credit limit",
            ]

            with open(patterns_file, "w") as f:
                for pattern in patterns:
                    f.write(pattern + "\n")

        return patterns_file

    def train(self, num_epochs=3, batch_size=8):
        """Fine-tune model on fraud patterns."""
        print("=" * 60)
        print("FRAUD-GPT TRAINING")
        print("=" * 60)

        # Load base model
        print(f"\n[1/4] Loading base model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Apply LoRA
        print("[2/4] Applying LoRA configuration...")
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare data
        print("[3/4] Preparing training data...")
        train_file = self.prepare_training_data()

        dataset = TextDataset(
            tokenizer=tokenizer, file_path=str(train_file), block_size=128
        )

        # Train
        print("[4/4] Fine-tuning model...")
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10,
            save_total_limit=2,
            logging_steps=10,
            no_cuda=self.device == "cpu",
        )

        trainer = Trainer(
            model=model, args=training_args, train_dataset=dataset, data_collator=None
        )

        trainer.train()

        # Save final model
        output_path = self.output_dir / "fraud_gpt.pt"
        torch.save(model.state_dict(), output_path)
        print(f"\n✓ Saved: {output_path}")

        print("\n" + "=" * 60)
        print("FRAUD-GPT TRAINING COMPLETE")
        print("=" * 60)

        return model, tokenizer


if __name__ == "__main__":
    trainer = FraudGPTTrainer(model_name="gpt2")
    model, tokenizer = trainer.train(num_epochs=3, batch_size=8)
