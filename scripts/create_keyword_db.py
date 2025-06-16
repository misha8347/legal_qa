#load parent folder
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

#load libraries
from src.retrieval.keyword_db import KeywordDB
import pandas as pd
from loguru import logger

def main(path_to_data: str):
    df = pd.read_csv(path_to_data)

    keyword_db = KeywordDB()
    keyword_db.create_knowledge_base(df)

if __name__ == "__main__":
    path_to_data = '/hdd/mikhail/projects/egov_knowledge_base/data/df_combined_with_summaries.csv'
    main(path_to_data)