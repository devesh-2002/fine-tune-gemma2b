import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import wandb
from huggingface_hub import notebook_login

notebook_login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "google/gemma-2b-pytorch"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

ds = load_dataset("tuneai/oasst2_top1_chatgpt_format")

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

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

wandb.login()
wandb.init(project="ml-assignment-tuneai", name="my-first-run")

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset['train'],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=4,
        push_to_hub=True,
        output_dir="fine-tune-gemma",
        optim="paged_adamw_8bit",
        report_to="wandb",
        logging_dir="./logs",
    ),
    peft_config=lora_config,
    dataset_text_field="text"
)

trainer.train()

wandb.finish()