# Always name it rag.py
# Requirements:
# pip install sentence-transformers faiss-cpu rank-bm25 transformers torch numpy scikit-learn tqdm regex
# install faiss-cpu if mac M1, faiss-gpu if you have a CUDA GPU

import argparse, json, os, pickle, re, sys, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer   # from sentence-transformers/all-MiniLM-L6-v2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#################################################################################
# Initial Chunk Data Input Functions
#################################################################################
def read_jsonl(path: str) -> List[Dict]:
    '''Read a JSONL file (e.g. chunks.clean.jsonl) and return a list of dictionaries.'''
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_chunk_text(row: Dict) -> str:
    '''
    Return the retrieval text for a chunk.
    Prefer: title + text/content. Avoid IDs/URLs leaking into BM25.
    '''
    title = row.get("title")
    parts = []      # pieces to join
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())

    content = row.get("text", None)
    if isinstance(content, str) and content.strip():
        c = content.strip()
        if title and title.strip() in c:
            parts = [c]    # title already present in content, only keep content
        else:
            parts.append(c)

    # Join + whitespace cleanup
    merged = "\n\n".join(parts).strip()
    merged = re.sub(r"\s+", " ", merged)
    return merged


def get_chunk_meta(row: Dict) -> Dict:
    '''Keep lightweight, display-only fields (do NOT feed into BM25).'''
    return {
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "title": (row.get("title") or "").strip() or None,
        "url": (row.get("url") or "").strip() or None,
    }


#################################################################################
# Sparse (BM25) Index
#################################################################################
@dataclass
class BM25Index:
    # BM25 scorer from rank-bm25 to build over a tokenized corpus
    bm25: BM25Okapi
    # tokenized corpus: a list of chunks, where each chunk is a list of token strings.
    tokenized_docs: List[List[str]]


    @staticmethod
    def normalize_text_for_bm25(s: str) -> List[str]:
        '''
        Simple text normalization: lowercase + keep words/numbers.
        Finds all sequences of one or more lowercase letters and digits in `s`
        (after lowercasing) and returns them as a list of strings.
        '''
        return re.findall(r"[a-z0-9]+", s.lower())


    @classmethod
    def build(cls, texts: List[str]) -> "BM25Index":
        '''Tokenizes every chunk (normalize_text_for_bm25) and builds BM25Okapi'''
        tokenized = [cls.normalize_text_for_bm25(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, tokenized_docs=tokenized)


    def retrieve(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Tokenizes the query, computes BM25 scores over all chunks.
        Returns: the top-k chunk indices + their scores
        '''
        if k == 0:     # top-0 requested
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        
        q_tok = self.normalize_text_for_bm25(query)  # query tokens

        # Handle empty queries
        if not q_tok:     # query is empty after normalization
            n = len(self.tokenized_docs)
            k = min(k, n)
            idx = np.arange(k, dtype=int)
            return idx, np.zeros(k, dtype=np.float32)

        scores = np.asarray(self.bm25.get_scores(q_tok), dtype=np.float32)  # 1 score per chunk
        n = scores.shape[0]
        k = min(k, n)

        # Efficient top-k: argpartition, then sort those k by score desc
        topk_unsorted = np.argpartition(scores, -k)[-k:]
        topk_sorted = topk_unsorted[np.argsort(scores[topk_unsorted])[::-1]]
        return topk_sorted, scores[topk_sorted]


#################################################################################
# Dense (FAISS) Index
#################################################################################
@dataclass
class DenseIndex:
    # Hugging Face SentenceTransformer model id
    model_name: str
    # Embedding dimensionality: 384 for MiniLM-L6-v2
    dim: int
    # FAISS index holding the document vectors for fast similarity search
    index: faiss.Index
    # Choose to L2-normalized embeddings or not (True: inner product = cosine)
    normalize: bool
    # Optional: if we have cached the encoder for reuse
    _st: Optional[SentenceTransformer] = field(default=None, init=False, repr=False, compare=False)
    

    @property
    def st(self) -> SentenceTransformer:
        '''Auto-cache the encoder on first access'''
        if self._st is None:
            self._st = SentenceTransformer(self.model_name)
        return self._st


    @staticmethod
    def build(texts: List[str], model_name: str, normalize: bool = True, batch: int = 256) -> "DenseIndex":
        '''
        Uses SentenceTransformer embedder to encode each chunk (one item in texts) into a vector.
        Builds a FAISS IndexFlatIP (inner product). 
        If vectors are L2-normalized (normalize=True), inner product == cosine similarity.
        Reason: Implements the required dense retriever using embeddings and FAISS.
        '''
        st = SentenceTransformer(model_name)
        # Encodes all the N chunks to a NumPy array embeddings of shape (N, dim)
        embeddings = st.encode(texts, 
                               batch_size=batch, 
                               convert_to_numpy=True, 
                               show_progress_bar=True, 
                               normalize_embeddings=normalize)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))   # FAISS requires float32
        
        # Wrap the index and metadata into the DenseIndex dataclass
        dense_index = DenseIndex(model_name=model_name, dim=dim, index=index, normalize=normalize)
        # Optionally keep embeddings in memory for saving later. Comment out if memory is tight
        dense_index._embs = embeddings
        dense_index._st = st
        return dense_index


    def retrieve(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Encodes the query with the same embedder, then runs faiss.index.search 
        to return top-k dense neighbors (indices, scores).
        '''
        if not hasattr(self, "_st") or self._st is None:
            self._st = SentenceTransformer(self.model_name)
        # Encode the query to a (1, dim) vector
        q_enc = self._st.encode([query], convert_to_numpy=True, normalize_embeddings=self.normalize)
        S, I = self.index.search(q_enc.astype(np.float32), k)
        # top-k indices: np.ndarray[int], top-k scores: np.ndarray[float32]
        return I[0], S[0]


#################################################################################
# Weighted Fusion
#################################################################################
def minmax(arr: np.ndarray) -> np.ndarray:
    '''
    Normalize scores from different retrievers onto the same [0, 1] scale.
    Element-wise calculate: (x − min x)/(max x − min x).
    If all scores are equal, return all zeros.
    '''
    min_, max_ = np.min(arr), np.max(arr)
    if max_ <= min_:
        return np.zeros_like(arr)
    return (arr - min_) / (max_ - min_)


def zscore(arr: np.ndarray) -> np.ndarray:
    '''Z-score normalization: (x - mean) / (stddev + 1e-8)'''
    mu, sigma = np.mean(arr), np.std(arr) + 1e-8
    return (arr - mu) / sigma


def hybrid_weighted(k: int, bm25_idx: BM25Index, dense_idx: DenseIndex, query: str, 
                    alpha: float = 0.5, pool: int = 50) -> Tuple[List[int], np.ndarray]:
    '''
    Weighted fusion of sparse (BM25) & dense (embeddings/FAISS) 
    by score normalization & weighted sum.
    alpha: try {0.3, 0.5, 0.7}.
    pool: bigger pools (50-100) increase chances of finding 
    the right doc after fusion but adds to compute
    '''
    # Retrieve a large pool of chunks (i.e. docs) from each method, normalize, then fuse
    b_idx, b_scores = bm25_idx.retrieve(query, pool)
    d_idx, d_scores = dense_idx.retrieve(query, pool)

    # Unify chunk (i.e. doc) candidate sets
    doc_ids = sorted(set(b_idx.tolist()) | set(d_idx.tolist()))
    id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    b_vec = np.zeros(len(doc_ids), dtype=np.float32)
    d_vec = np.zeros(len(doc_ids), dtype=np.float32)
    for i, s in zip(b_idx, b_scores): b_vec[id_to_pos[i]] = s
    for i, s in zip(d_idx, d_scores): d_vec[id_to_pos[i]] = s

    # Normalize then fuse using a weighted sum
    b_norm = minmax(b_vec)
    d_norm = minmax(d_vec)
    fused = alpha * d_norm + (1 - alpha) * b_norm

    # Sort by fused score and choose the top-k post fusion
    order = np.argsort(fused)[::-1][:k]
    return [doc_ids[i] for i in order], fused[order]


#################################################################################
# Reciprocal Rank Fusion (RRF)
#################################################################################
def rank_of(targets, x):
    '''
    Calculate the rank of x in the targets list.
    Targets is the list of retrieved IDs already in ranked order (best first).
    Missing docs get a huge rank (effectively near-zero contribution later).
    '''
    pos = {doc_id: r for r, doc_id in enumerate(targets)}  # 0-based
    return pos.get(x, 10_000)  # large if missing


def hybrid_rrf(k: int, bm25_idx: BM25Index, dense_idx: DenseIndex, query: str, 
               pool: int = 50, K: int = 60) -> Tuple[List[int], np.ndarray]:
    '''
    Reciprocal Rank Fusion (RRF) of BM25 + dense (FAISS) scores.
    Combine ranks instead of raw scores. 
    It’s robust when score scales are incomparable.
    K: try 60. It is a dampening factor to avoid over-weighting the very top items.
    Lower K makes top ranks more dominant; higher K flattens contributions.
    '''
    b_idx, _ = bm25_idx.retrieve(query, pool)
    d_idx, _ = dense_idx.retrieve(query, pool)

    # Unify chunk (i.e. doc) candidate sets
    doc_ids = sorted(set(b_idx.tolist()) | set(d_idx.tolist()))
    
    scores = []
    for doc in doc_ids:
        rb = rank_of(b_idx, doc)
        rd = rank_of(d_idx, doc)
        # each of the 2 retrievers contributes 1 / (K + rank + 1)
        scores.append(1 / (K + rb + 1) + 1 / (K + rd + 1))
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in order], scores[order]




#################################################################################
# Arg Parsing
#################################################################################
def main():
    p = argparse.ArgumentParser(description="Framework-free RAG baseline (BM25 + FAISS + Hybrid + HF reader)")
    sp = p.add_subparsers()

    p_idx = sp.add_parser("index", help="Build BM25 + FAISS over chunks")
    p_idx.add_argument("--chunks", required=True)
    p_idx.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p_idx.add_argument("--batch", type=int, default=256)
    p_idx.add_argument("--outdir", default="artifacts/index")
    p_idx.set_defaults(func=cmd_index)

    p_ans = sp.add_parser("answer", help="Run retrieval + reader and write system_output_test.json")
    p_ans.add_argument("--questions", required=True)
    p_ans.add_argument("--indexdir", default="artifacts/index")
    p_ans.add_argument("--reader", default="mistralai/Mistral-7B-Instruct-v0.3")  # choose any HF open-weight model ≤32B
    p_ans.add_argument("--k", type=int, default=8)
    p_ans.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default="hybrid")
    p_ans.add_argument("--fusion", choices=["weighted", "rrf"], default="weighted")
    p_ans.add_argument("--alpha", type=float, default=0.5)  # weight for dense in weighted fusion
    p_ans.add_argument("--pool", type=int, default=50)
    p_ans.add_argument("--rrfK", type=int, default=60)
    p_ans.add_argument("--max_new_tokens", type=int, default=64)
    p_ans.add_argument("--temp", type=float, default=0.0)
    p_ans.add_argument("--system_out", default="data/test/system_output_test.json")
    p_ans.set_defaults(func=cmd_answer)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()

