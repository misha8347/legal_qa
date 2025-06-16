import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
sys.path.append(parent_dir)

from sklearn.feature_extraction.text import TfidfVectorizer
from ollama import chat, ChatResponse
from src.retrieval.hybrid_retriever import HybridRetriever
import numpy as np
from nltk import word_tokenize
import nltk
import time
from loguru import logger
nltk.download('stopwords')

sw_rus = nltk.corpus.stopwords.words('russian')

def greed_sum_query_optimized(text, query, min_df=0.043, max_df=1.0, stop_words=sw_rus):
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words)
    X = vectorizer.fit_transform(text)
    
    # Get the vocabulary and map it to indices
    vocab_index = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}
    
    # Emphasize query tokens in the vectorized matrix
    query_tokens = word_tokenize(query.lower())
    cols_to_emphasize = [vocab_index[q] for q in query_tokens if q in vocab_index]
    
    # Create an emphasis multiplier matrix
    emphasis_matrix = np.ones(X.shape[1])
    emphasis_matrix[cols_to_emphasize] *= 50
    
    # Apply emphasis by element-wise multiplication
    X = X.multiply(emphasis_matrix)
    
    # Convert to dense array if needed for further operations
    X = X.toarray()

    # Initialize list to keep track of selected sentence indices
    selected_indices = []
    
    # Iteratively select sentences
    for _ in range(len(text)):
        # Compute the sum of TF-IDF scores for each sentence
        sentence_scores = X.sum(axis=1)
        
        # Select the sentence with the maximum score
        best_index = np.argmax(sentence_scores)
        selected_indices.append(best_index)
        
        # Remove the selected sentence's terms from consideration
        selected_terms = np.nonzero(X[best_index])[0]
        X[:, selected_terms] = 0  # Zero out selected terms across all sentences

        # Stop if all terms have been exhausted
        if np.sum(X) == 0:
            break
    
    # Sort indices to maintain original order of sentences
    selected_indices.sort()
    
    # Return the selected sentences as a summary
    summary = [text[i] for i in selected_indices]
    return summary

class QAPipeline:
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid_retriever = hybrid_retriever

    def generate_response(self, query: str, top_n: int = 10):
        logger.info(f'query: {query}')
    
        start = time.time()
        results = self.hybrid_retriever.hybrid_search(query, top_n=top_n)
        logger.info(f"Hybrid search took {time.time() - start:.2f} seconds")

        # сбор саммари результатов относительно запроса
        start = time.time()
        final_sum = []
        for i, (doc_id, text, score) in enumerate(results):
            # print(f"\n[{i+1}] ID: {doc_id} | Score: {score:.4f}")
            # print(text[:500] + "...")
            summary = ' '.join(greed_sum_query_optimized(nltk.sent_tokenize(text), query, min_df=0.043, max_df=1.0))
            final_sum.append(summary)
            logger.info(len(text))
        final_sum = ' '.join(final_sum)
        logger.info(f"length of the final summary: {len(final_sum)}")
        logger.info(f"Summarization relative to the query took {time.time() - start:.2f} seconds")

        try:
            start = time.time()
            prompt = f"""
            Вопрос: {query}\n\nТекст для преобразования в связный ответ с разбиением на параграфы и генерацией списков:\n\n{final_sum}. Отвечай на русском языке. Любые включения на других языках должны быть переведены на русский язык. Транслит должен быть преобразован в кириллицу. Если в предоставленном тексте нет ответа на вопрос, напиши об этом фразой 'Предоставленный текст не содержит ответа на вопрос', и попытайся ответить на на вопрос самостоятельно, предоставив ссылки на источники. Но не выдумывай ответ, используй только действительные факты.
            """

            response: ChatResponse = chat(
                model='llama3.2:3b',
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in producing legal recommendations based on given content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                options={
                    "temperature": 0.2,  # more deterministic
                    "num_predict": 1024  # number of tokens to predict
                }
            )
            logger.info(f"LLM response took {time.time() - start:.2f} seconds")
            return response.message.content
        
        except Exception as e:
            return "К сожалению, выполнить сглаживание текста не удалось по техническим причинам. Прилагаем необработанный текст: \n\n" + final_sum
