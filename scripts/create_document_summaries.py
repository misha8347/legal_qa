#load parent folder
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

#import libraries
from loguru import logger
from tqdm import tqdm
import pandas as pd
from src.summarization import greed_sum
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt_tab')


model = SentenceTransformer('ai-forever/sbert_large_nlu_ru')

def get_summary_vector(doc_text):
    sentences = nltk.sent_tokenize(doc_text)
    summary = greed_sum(sentences, num_sent=0)  # без ограничения num_sent
    distilled_text = ' '.join(summary)
    return model.encode(distilled_text)

def main(df: pd.DataFrame, path_to_save_data: str):
    assert 'text' in df
    
    logger.info(f'starting summarization of {len(df)} records')

    summarized_texts = []
    for index, value in tqdm(df.iterrows(), total=len(df)):
        if isinstance(value['text'], str):
            sentences = nltk.sent_tokenize(value['text'])
            summary = greed_sum(sentences, num_sent=0, min_df=0.02)
        else:
            print(f"Skipping invalid text: {value['text']}")
            summary = ''

        summarized_texts.append(summary)
    
    df['text_summary'] = summarized_texts
    logger.info(f'saving {len(summarized_texts)} summaries')

    df.to_csv(path_to_save_data, index=False)
    logger.info(f'dataset with summaries saved successfully!')
    

if __name__ == "__main__":
    path_to_data = '../data/df_combined.csv'
    path_to_save_data = '../data/df_combined_with_summaries.csv'
    df = pd.read_csv(path_to_data)
    main(df, path_to_save_data)