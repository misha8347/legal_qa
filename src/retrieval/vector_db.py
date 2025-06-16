import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from loguru import logger
import os
from langchain_core.documents import Document
import torch
import json
from tqdm import tqdm
from typing import List, Tuple, Dict


class VectorDB:
    def __init__(self, 
                 db_path: str = '/Users/mikhailogay/Documents/MaProjects/egov_knowledge_base/data/vector_db'):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)

        self.index_path = os.path.join(self.db_path, "vector.index")
        self.metadata_path = os.path.join(self.db_path, "vector_meta.json")

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = SentenceTransformer('ai-forever/sbert_large_nlu_ru').to(self.device)


    def create_knowledge_base(self, df: pd.DataFrame):
        assert 'text_summary' in df
        assert 'id' in df
        assert 'url' in df
        logger.info('starting to create vector database')


        df = df[df['text'].fillna('').str.strip() != '']

        self.doc_ids = df['id'].tolist()
        self.urls = df['url'].tolist()
        self.text_summaries = df['text_summary'].tolist()

        embeddings = self.embedding_model.encode(self.text_summaries, convert_to_numpy=True)
        d = self.embedding_model.get_sentence_embedding_dimension()

        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, 100, 16, 8)  # 100 центроидов, 16 субвекторов, по 8 битов каждый
        self.index.train(embeddings)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

        json.dump({"ids": self.doc_ids, "urls": self.urls, "text_summaries": self.text_summaries}, open(self.metadata_path, "w", encoding="utf-8"), ensure_ascii=False)
        logger.info('vector database created successfully!')

    def load_db(self):
        self.index = faiss.read_index(self.index_path)
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.doc_ids = self.metadata['ids']
        self.text_summaries = self.metadata['text_summaries']
        logger.info('vector database loaded successfully!')


    def search(self, query, top_k=5) -> List[Dict]:
        # Получение эмбеддинга запроса
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")

        # Поиск в FAISS
        D, I = self.index.search(query_vector, top_k)

        # Возврат результатов
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.doc_ids):
                results.append({"id": self.doc_ids[idx], "score": float(score), "text_summary": self.text_summaries[idx]})
        return results