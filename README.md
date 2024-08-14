# Tune AI MLE Assignment
## Setup and run the project
1. Fork and Clone the repository
2. Copy `.env.example` to `.env`
3. Add the keys to the .env (HuggingFace Token and WandB Api Key)
   
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
