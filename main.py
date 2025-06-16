from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline.qa_pipeline import QAPipeline
from src.retrieval.hybrid_retriever import HybridRetriever

# ML
hybrid_retriever = HybridRetriever()
qa_pipeline = QAPipeline(hybrid_retriever)

# Backend
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/generate_legal_qa_response")
async def generate_legal_qa_response(query: str):
    response = qa_pipeline.generate_response(query)
    return {'response': response}

