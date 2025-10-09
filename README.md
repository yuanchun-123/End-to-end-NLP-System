# Retrieval Augmented Generation (RAG) for Pittsburgh Facts

## 1. Project Pipeline

### 1. Data Preparation and Initial System
Goal: Crawl/scrape public pages for Pittsburgh/CMU, clean HTML/PDF into plain text, and produce boundary-aware chunks (with lead/infobox micro-chunks) for indexing.

Code File: 
- ```nlp_data.ipynb```: data scraping and chunking

Inputs: 
- ```seeds.txt```: root domains and high-signal pages.
- Raw downloaded HTML/PDF files stored in data/raw/

Outputs:
- ```data/corpus/documents.jsonl```: cleaned text documents (title, URL, metadata).
- ```data/corpus/chunks.jsonl```:  boundary-aware chunks with overlap.
- ```data/corpus/chunks.clean.jsonl```: filtered version removing low-text and noisy pages.
- ```data/corpus/chunks.with_dates.jsonl```: same chunks with temporal tags (for event questions).


### 2. RAG Models Implementation

Goal: Implement a RAG architecture that combines a sparse, a dense, and two hybrid retriever components with a question-answering LLM reader to effectively utilize both retrieved context and generative capabilities. The model takes in cleaned chunks from step 1 as the knowledge base for retrieval.

Code Files: 
- ```implementation_evaluation.ipynb```: command-line like file for RAG pipeline, including indexing, retrieval, and answer generation
- ```rag.py```: backend modules and methods for indexing, retrieval, and answer generation

Inputs: 
- ```data/corpus/chunks.with dates.jsonl```: cleaned chunks from step 1
- ```data/test/questions.txt```: questions
- ```data/test/reference answers.json```: answers to the questions

Outputs: 
- ```index/``` folder: contains pickle and json files from the indexing step
- ```data/test/``` + ```system_output_dense.json```, ```system_output_sparse.json```, ```system_output_test.json```: outputs from testing on the test dataset for dense, sparse, and hybrid retrievers, respectively

### 3. Evaluation Results \& Analysis

Goal: Comparing sparse, dense, and hybrid pipelines; assess retrieval quality, answer accuracy, latency, and error modes; provide ablations and a final recommendation.

Code Files: Same as Step 2

Inputs:

- ```data/test/system_output_dense.json```
- ```data/test/system_output_sparse.json```
- ```data/test/system_output_test.json```

Outputs:
- ```index/``` folder: contains pickle and json files from the indexing step
- ```system_outputs/system_output_1.json```

## 2. Steps to Run the Codes

### 1. Install Required Packages
```pip install sentence-transformers rank-bm25 transformers torch numpy scikit-learn tqdm regex```
```pip install faiss-cpu``` if you're on CPU or mac M1. ```pip install faiss-gpu``` or ```faiss-gpu-cu12``` if you have a CUDA GPU. Only do one of these two.

### 2. Run RAG Model
Run the notebook file ```implementation_evaluation.ipynb```.
Make sure that you have ```rag.py``` in the same directory.








