
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llm import format_diagnose, index_diagnose_db, chat, fill_schema
from dto import DiagnoseRequest, PlaygroundRequest, ChatRequest, GraphRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL in production for security
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Add OPTIONS to allowed methods
    allow_headers=["*"],
)


# routes not protected

@app.get("/")
async def root():
    return {"message": "hello"}

@app.post("/index/diagnose/{id}")
async def diagnose(id: str, diagnose: DiagnoseRequest):
    index_diagnose_db(id, diagnose)
    return { "message" : "success"}