# Fine Tuned Gemma-2b-pytorch
We focus on fine-tuning a pre-trained language model and deploying it as a web service. The main steps involved are:

1. **Fine-Tuning a Pre-trained Model**: We use the `google/gemma-2b-pytorch` model and fine-tune it on a chat dataset (`tuneai/oasst2_top1_chatgpt_format`). 

2. **Creating a Server**: We deploy the fine-tuned model using FastAPI and Uvicorn, creating an API that adheres to the OpenAI chat completion API specification.

## Setup and run the project
1. Fork and Clone the repository
2. Copy `.env.example` to `.env`
3. Add the keys to the .env (HuggingFace Token and WandB Api Key)


If you want to run both server and trainer, then you can run using shell script :
```
./run.sh
```
   
If you want to manually run :

4. Create a virtual env and activate it
```
python -m venv venv
source venv/bin/activate
```
5. Install the requirements :
```
pip install -r requirements.txt
```
6. If you want to train the model :
```
python trainer.py train --batch_size=1 --grad_accum_steps=4 --warmup_steps=10 --max_steps=100 --learning_rate=1e-5
```
7. If you want to run the server :
```
uvicorn server:app
```

### Example Request

Use the following `curl` command to send a request:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "human", "content": "I need help identifying a bolt thread. The hardware store is closed and the only tool I have is a ruler."},
  ]
}'
```
## Learnings and Findings

1. **Fine-Tuning**: The process of fine-tuning allowed us to adapt a large pre-trained PyTorch model to a specific dataset, improving its performance on conversational data. Key considerations included selecting appropriate hyperparameters and managing computational resources.

2. **Configuration Management**: PyTorch models often require configuration files to ensure that the model is loaded with the correct settings. Proper management of these configurations is essential for accurate model operation and fine-tuning.

3. **Deployment**: Setting up a FastAPI server for the model demonstrated the ease of integrating machine learning models into web applications.

