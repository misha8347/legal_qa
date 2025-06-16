#load parent folder
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
sys.path.append(parent_dir)

#load libraries
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.utils.utils import load_obj
from nltk.tokenize import word_tokenize
import json
from loguru import logger

class HybridRetriever:
    def __init__(
        self,
        bm25_path: str = '/Users/mikhailogay/Documents/MaProjects/egov_knowledge_base/data/keyword_db/bm25',
        vector_index_path: str = '/Users/mikhailogay/Documents/MaProjects/egov_knowledge_base/data/vector_db/vector.index',
        metadata_path: str = '/Users/mikhailogay/Documents/MaProjects/egov_knowledge_base/data/vector_db/vector_meta.json'
    ):
        self.bm25 = load_obj(bm25_path)
        self.model = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
        self.index = faiss.read_index(vector_index_path)

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.doc_ids = self.metadata['ids']
        self.text_summaries = self.metadata['text_summaries']
        logger.info('loaded hybrid retriever successfully!')

    # === Поиск ===
    def hybrid_search(self, query, top_n=10, alpha=0.5):
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        _, faiss_indices = self.index.search(query_vec, len(self.doc_ids))

        faiss_scores = np.zeros(len(self.doc_ids))
        for rank, idx in enumerate(faiss_indices[0]):
            faiss_scores[idx] = 1.0 - (rank / len(self.doc_ids))  # нормализация ранга

        # Гибридное объединение
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores

        # Топ-N результатов
        top_ids = np.argsort(hybrid_scores)[::-1][:top_n]
        # results = [(self.doc_ids[i], self.text_summaries[i], hybrid_scores[i]) for i in top_ids]

        seen_summaries = set()
        results = []

        for i in top_ids:
            summary = self.text_summaries[i]
            if summary in seen_summaries:
                continue
            seen_summaries.add(summary)
            results.append((self.doc_ids[i], summary, hybrid_scores[i]))
        
        del seen_summaries
        return results