import os
import json
import torch
from torch.utils.data import Dataset
import transformers
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from dataclasses import dataclass, field
from typing import Optional, Dict

# Data class to hold model-specific arguments
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base", metadata={"help": "Pretrained model name or path"})

# Data class to hold arguments related to data processing
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

# Custom training arguments extending from transformers' TrainingArguments
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the pre-trained models downloaded from s3"})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use for training"})
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA in the model."})

# Custom Dataset for supervised fine-tuning
class SupervisedDataset(Dataset):
    """Dataset class for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer, model_max_length, user_tokens=[195], assistant_tokens=[196]):
        """
        Initialize the SupervisedDataset.

        Args:
            data_path: Path to the data file.
            tokenizer: Tokenizer to use for encoding the texts.
            model_max_length: Maximum length of the model input.
            user_tokens: List of token IDs representing user inputs. Defaults to [195].
            assistant_tokens: List of token IDs representing assistant replies. Defaults to [196].
        """
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))  # Load data from the specified path
        self.tokenizer = tokenizer  # Set the tokenizer
        self.model_max_length = model_max_length  # Set the maximum sequence length for the model
        self.user_tokens = user_tokens  # Set the token IDs for user inputs
        self.assistant_tokens = assistant_tokens  # Set the token IDs for assistant replies
        self.ignore_index = -100  # Set the ignore index for the labels

        # Preprocess the first element to check everything is set correctly
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))  # Decode and print the input IDs
        labels = [id_ for id_ in item["labels"] if id_ != self.ignore_index]  # Extract valid labels
        print("label:", self.tokenizer.decode(labels))  # Decode and print the labels

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def preprocessing(self, example):
        """
        Preprocess a single data example.

        Args:
            example: A single example from the dataset.

        Returns:
            A dictionary containing processed input IDs, labels, and attention mask.
        """
        input_ids = []
        labels = []

        # Process each message in the conversation
        for message in example["conversations"]:
            from_ = message["from"]  # Source of the message (human or assistant)
            value = message["value"]  # Content of the message
            value_ids = self.tokenizer.encode(value)  # Encode the content into IDs

            # Process input IDs and labels based on the message source
            if from_ == "human":
                input_ids += self.user_tokens + value_ids  # Add user tokens and value IDs for human messages
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)  # Set labels for user messages
            else:
                input_ids += self.assistant_tokens + value_ids  # Add assistant tokens and value IDs for assistant messages
                labels += [self.ignore_index] + value_ids  # Set labels for assistant messages

        # Add end-of-sequence token at the end of the sequence
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)

        # Truncate or pad input_ids and labels to model_max_length
        input_ids = input_ids[:self.model_max_length] + [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels = labels[:self.model_max_length] + [self.ignore_index] * (self.model_max_length - len(labels))

        # Convert lists to PyTorch tensors
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)

        # Create an attention mask (0 for padded positions and 1 for the rest)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Return a dictionary containing the input IDs, labels, and attention mask
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Return the preprocessed item at the specified index
        return self.preprocessing(self.data[idx])

# Main training function
def train():
    # Parse arguments using Hugging Face's argument parser
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load the pre-trained model and tokenizer from the specified path
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, cache_dir=training_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, trust_remote_code=True, model_max_length=training_args.model_max_length, cache_dir=training_args.cache_dir)

    # If LoRA is used, modify the model accordingly
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        # Define LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Initialize the dataset and trainer
    dataset = SupervisedDataset(data_args.data_path, tokenizer, training_args.model_max_length)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)

    # Start training and save the model and trainer state
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

# Entry point for the script
if __name__ == "__main__":
    train()

