#load parent folder
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

#import libraries
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from utils.utils import load_obj, save_obj
from rank_bm25 import BM25Okapi
import nltk
import json
from loguru import logger
nltk.download('punkt')
from nltk.tokenize import word_tokenize


class KeywordDB:
    def __init__(self,
                 db_path: str = '/Users/mikhailogay/Documents/MaProjects/egov_knowledge_base/data/keyword_db'):
        
        os.makedirs(db_path, exist_ok=True)
        self.save_path = os.path.join(db_path, 'bm25')
        self.metadata_path = os.path.join(db_path, 'keyword_meta.json')

    def create_knowledge_base(self, df: pd.DataFrame):
        assert 'text_summary' in df
        logger.info('starting to create keyword database')

        df = df[df['text'].fillna('').str.strip() != '']

        self.doc_ids = df['id'].tolist()
        self.urls = df['url'].tolist()
        self.text_summaries = df['text_summary'].tolist()

        self.bm25 = BM25Okapi([word_tokenize(d) for d in tqdm(self.text_summaries)])
        save_obj(self.bm25, self.save_path)
        
        json.dump({"ids": self.doc_ids, "urls": self.urls, "text_summaries": self.text_summaries}, open(self.metadata_path, "w", encoding="utf-8"), ensure_ascii=False)
        logger.info('keyword database created successfully!')

    def load_db(self):
        self.bm25 = load_obj(self.save_path)

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.doc_ids = self.metadata['ids']
        self.text_summaries = self.metadata['text_summaries']
        logger.info('Keyword database loaded successfully!')

    def search(self, query, top_k=5) -> List[Dict]:
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        # Получаем топ-результаты
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Выводим
        results = []
        for i in top_indices:
            result = {
                'id': self.doc_ids[i],
                'score': scores[i],
                'text_summary': self.text_summaries[i]
            }

            results.append(result)
        
        return results
