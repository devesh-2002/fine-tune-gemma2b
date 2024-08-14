from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nest_asyncio
import uvicorn

class Message(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(..., example="I need help identifying a bolt thread.")

class ChatCompletionRequest(BaseModel):
    messages: list[Message]

class Choice(BaseModel):
    message: Message

class ChatCompletionResponse(BaseModel):
    choices: list[Choice]

app = FastAPI()

model_name = "devesh-2002/fine-tuned-gemma"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        messages = request.messages

        conversation = ""
        for message in messages:
            role = message.role
            content = message.content
            conversation += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

        inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=512,max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = {
            "choices": [{"message": {"role": "gpt", "content": response_text}}]
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
