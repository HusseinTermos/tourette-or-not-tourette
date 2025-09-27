from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/score")
async def score(request: Request):
    # Read and ignore the audio; always reply 0.4
    await request.body()
    return {"score": 0.4}
