
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llm import format_diagnose, index_diagnose_db, chat, fill_schema
from dto import DiagnoseRequest, PlaygroundRequest, ChatRequest, GraphRequest

app = FastAPI()



# routes not protected

@app.get("/")
async def root():
    return {"message": "hello"}