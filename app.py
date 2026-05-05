from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
# ✅ ONLY ONE app instance
app = FastAPI()

# ✅ ADD CORS IMMEDIATELY HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # VERY IMPORTANT
    allow_headers=["*"],
)
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
import os

# -------------------------------
# 1. Load data
# -------------------------------
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

# -------------------------------
# 2. Split
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# -------------------------------
# 3. Embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# 4. FAISS
# -------------------------------
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 5. Groq LLM
# -------------------------------
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# -------------------------------
# 6. FastAPI
# -------------------------------

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/chat")
def chat(q: Query):
    docs = retriever.invoke(q.question)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Context:
{context}

Question:
{q.question}
"""

    answer = ask_llm(prompt)
    return {"answer": answer}

from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()