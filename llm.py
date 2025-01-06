from dto import DiagnoseRequest
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")

faiss_dbs = {}

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_KEY)
embeddings = OpenAIEmbeddings(
    openai_api_key = OPENAI_KEY
)



document_format = """
The doctor {docter_name} recorded following details on date {date}\n
SYMPTOMS:
{symptoms}
DIAGNOSIS:
{diagnosis}
MEDICATIONS:
{medications}
MEDICAL TESTS:
{medical_tests}
"""

def format_diagnose(diagnose: DiagnoseRequest):
    doc = document_format.format(
        docter_name = diagnose.doctor,
        date = diagnose.input_date,
        symptoms = diagnose.symptoms,
        diagnosis = diagnose.diagnosis,
        medications = diagnose.medications,
        medical_tests = diagnose.medical_tests
    )

    return doc


def index_diagnose_db(patient_id: str, diagnose: DiagnoseRequest):
    path = f"./local_data/{patient_id}/faiss_index"
    formatted_diagnose = format_diagnose(diagnose)
    db_new = FAISS.from_texts([formatted_diagnose], embeddings)
    if os.path.exists(path):
         print(f"File exists. id: {patient_id}")
         db_old = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
         db_old.merge_from(db_new)
         print(f"merged db's. id: {patient_id}")
         shutil.rmtree(path)
         print(f"deleted old db. id: {patient_id}")
         db_old.save_local(path)
    else:
        print(f"File does not exist. id: {patient_id}")
        db_new = FAISS.from_texts([formatted_diagnose], embeddings)
        db_new.save_local(path)
        print(f"saved newly created db. id: {patient_id}")

    faiss_dbs[patient_id] = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"loaded to memory. id: {patient_id}")