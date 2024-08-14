import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import wandb
from peft import PeftModel
import os

HF_TOKEN = os.environ['HF_TOKEN'] 
WANDB_API_KEY = os.environ['WANDB_TOKEN']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device):
    try:
        config = AutoConfig.from_pretrained(model_path,token=HF_TOKEN)
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16,token=HF_TOKEN)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")

def load_and_format_dataset(dataset_name):
    ds = load_dataset(dataset_name)

    def formatting_func(example):
        formatted_conversation = ""
        for turn in example.get('conversations', []):
            if isinstance(turn, dict):
                if turn.get('from') == 'human':
                    formatted_conversation += f"<start_of_turn>user\n{turn.get('value', '')}<end_of_turn>\n"
                elif turn.get('from') == 'gpt':
                    formatted_conversation += f"<start_of_turn>model\n{turn.get('value', '')}<end_of_turn>\n"
        return {"text": formatted_conversation.strip()}

    formatted_dataset = ds.map(formatting_func)
    return formatted_dataset

def train(model_path="google/gemma-2b-pytorch", dataset_name="tuneai/oasst2_top1_chatgpt_format", output_dir="fine-tuning-gemma", batch_size=1, grad_accum_steps=8, warmup_steps=1, max_steps=100, learning_rate=2e-5):
    model = load_model(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path,token=HF_TOKEN)
    formatted_dataset = load_and_format_dataset(dataset_name)
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    try:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="fine-tuning-gemma", name="fine-tuning-gemma")

        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset['train'],
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=4,
                push_to_hub=True,
                output_dir=output_dir,
                optim="paged_adamw_8bit",
                report_to="wandb",
                logging_dir="./logs",
            ),
            peft_config=lora_config,
            dataset_text_field="text"
        )

        trainer.train()
        wandb.finish()

        save_and_push_model(output_dir)

    except Exception as e:
        print(f"An error occurred: {e}")

def save_and_push_model(output_dir):
    try:
        base_model = load_model("google/gemma-2b-pytorch", device)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-pytorch")

        merged_model = PeftModel.from_pretrained(base_model, output_dir)
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        merged_model.push_to_hub(output_dir, use_temp_dir=False)
        tokenizer.push_to_hub(output_dir, use_temp_dir=False)

    except Exception as e:
        raise RuntimeError(f"Error saving and pushing model: {e}")

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'save_and_push_model': save_and_push_model
    })
