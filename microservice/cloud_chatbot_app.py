from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import time

app = FastAPI()

# Define request model
class RequestModel(BaseModel):
    model_name: str
    user_input: str
    max_tokens: int

@app.post("/infer")
async def infer(request: RequestModel):
    try:
        start_time = time.time()
        response = ollama.chat(
            model=request.model_name,
            messages=[{'role': 'user', 'content': request.user_input}],
            options={'num_predict': request.max_tokens}
        )
        end_time = time.time()
        print(response)
        return {
            "model": request.model_name,
            "response": response['message']['content'],
            "latency": round(end_time - start_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

