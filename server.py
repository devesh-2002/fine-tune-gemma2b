from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import time
import os
from pyngrok import ngrok
import uvicorn
import nest_asyncio

app = FastAPI()

# Load environment variables
MODEL_PATH = "devesh-2002/fine-tuned-gemma"
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "100"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class Message(BaseModel):
    from_: str = Field(..., alias="from")
    value: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = Field(DEFAULT_MAX_TOKENS, ge=1)
    temperature: Optional[float] = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)

@app.post("/v1/chat/completions")
async def chat_completion(chat_request: ChatCompletionRequest):
    try:
        # Process the messages
        prompt = ""
        for msg in chat_request.messages:
            if msg.from_ == "human":
                prompt += f"Human: {msg.value}\n"
            elif msg.from_ == "gpt":
                prompt += f"Assistant: {msg.value}\n"
        prompt += "Assistant: "
        
        # Generate response
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + chat_request.max_tokens,
            temperature=chat_request.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
        
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        assistant_response = response_text.split("Assistant: ")[-1].strip()
        
        # Format the response to match the dataset format
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "from": "gpt",
                        "value": assistant_response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(tokenizer.encode(assistant_response)),
                "total_tokens": len(input_ids[0]) + len(tokenizer.encode(assistant_response))
            }
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
