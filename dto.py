from pydantic import BaseModel

class DiagnoseRequest(BaseModel):
    doctor: str
    symptoms: str
    diagnosis: str
    medications: str
    medical_tests: str
    input_date: str

class PlaygroundRequest(BaseModel):
    name: str

class ChatRequest(BaseModel):
    message: str