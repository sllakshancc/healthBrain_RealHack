from pydantic import BaseModel

class DiagnoseRequest(BaseModel):
    doctor: str
    symptoms: str
    diagnosis: str
    medications: str
    medical_tests: str
    input_date: str