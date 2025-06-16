Prepare data:

    1. Make folder structure as shown below and put datasets in data/ folder 
    
    P.S. You can upload only vector_db and keyword_db folders

    2. In src/retrieval/hybrid_retriever.py configure paths properly

<pre lang="markdown">
```plaintext
data/
├── keyword_db/
│   ├── bm25.pkl
│   └── keyword_meta.json
├── vector_db/
│   ├── vector.index
│   └── vector_meta.json
├── df_combined.csv
└── df_combined_with_summaries.csv

src/
scripts/
notebooks/
```
</pre>


Run ML service

    1. create venv using "python3 -m venv venv"
    2. activate venv
    3. run "pip install -r requirements.txt"
    4. execute "run.sh" file