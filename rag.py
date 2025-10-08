#################################################################################
# rag.py
# Baseline RAG (BM25 + FAISS + Hybrid + Hugging Face reader)
# Data layout expected:
#   data/corpus/chunks.clean.jsonl
#   data/test/questions.txt
#   data/test/reference_answers.json

# Requirements:
# pip install sentence-transformers rank-bm25 transformers torch numpy scikit-learn tqdm regex
# pip install faiss-cpu if you're on CPU or mac M1. pip install faiss-gpu if you have a CUDA GPU. Only one of these two.
#################################################################################

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
# Initial Chunk Data Loading Functions
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
    # Trim leading/trailing whitespace and return ""(false) if the string is whitespace-only
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())

    content = row.get("text", None)
    if isinstance(content, str) and content.strip():
        # avoid duplicating title if it's already in the content
        c = content.strip()
        if title and title.strip() in c:
            parts = [c]
        else:
            parts.append(c)

    # Join & whitespace cleanup
    merged = "\n\n".join(parts).strip()
    merged = re.sub(r"\s+", " ", merged)
    return merged


def get_chunk_meta(row: Dict) -> Dict:
    '''Extract lightweight, serializable metadata for logging/citations.'''
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
    # BM25 scorer from rank-bm25
    bm25: BM25Okapi
    # Tokenized corpus: a list of chunks, where each chunk is a list of token strings.
    tokenized_docs: List[List[str]]


    @staticmethod
    def normalize_text_for_bm25(s: str) -> List[str]:
        '''Simple text normalization: lowercase + keep words/numbers.'''
        return re.findall(r"[a-z0-9]+", s.lower())


    @classmethod
    def build(cls, texts: List[str]) -> "BM25Index":
        '''Tokenize every chunk (normalize_text_for_bm25) and build BM25Okapi'''
        tokenized = [cls.normalize_text_for_bm25(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, tokenized_docs=tokenized)


    def retrieve(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Tokenize the query, compute BM25 scores over all chunks.
        Return: the top-k chunk indices + their scores
        '''
        if k == 0:     # top-0 requested
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        
        q_tok = self.normalize_text_for_bm25(query)  # query tokens

        # Handle empty queries
        if not q_tok:
            n = len(self.tokenized_docs)    # total number of chunks
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
        # if already cached, reuse
        if not hasattr(self, "_st") or self._st is None:
            self._st = SentenceTransformer(self.model_name)
        # Encode the query to a (1, dim) vector
        q_enc = self._st.encode([query], convert_to_numpy=True, normalize_embeddings=self.normalize)
        S, I = self.index.search(q_enc.astype(np.float32), k)     # top-k scores, top-k indices: (1, k) arrays
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
    Weighted fusion of sparse (BM25) & dense (FAISS) by score normalization & weighted sum.
    alpha: try {0.3, 0.5, 0.7}.
    pool: bigger pools (50-100) increase chances of finding the right doc after fusion but larger compute.
    '''
    # Retrieve a large pool of chunks (i.e. docs) from each method
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
    targets: the list of retrieved IDs already in ranked order (best first).
    Missing docs get a huge rank 10,000 as penalty.
    '''
    pos = {doc_id: r for r, doc_id in enumerate(targets)}
    return pos.get(x, 10_000)


def hybrid_rrf(k: int, bm25_idx: BM25Index, dense_idx: DenseIndex, query: str, 
               pool: int = 50, K: int = 60) -> Tuple[List[int], np.ndarray]:
    '''
    Reciprocal Rank Fusion (RRF) of BM25 + dense (FAISS) scores.
    Combine ranks instead of raw scores.
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
# Reader (Hugging Face causal LM)
#################################################################################
class Reader:
    def __init__(self, model_name: str, max_new_tokens: int = 64, temperature: float = 0.0, include_sources: bool = False):
        '''Load a Instruct LLM from Hugging Face for generating answers.'''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the tokenizer for model_name from Hugging Face
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # Load the causal LM for generation
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.include_sources = include_sources


    def _format_prompt(self, question: str, passages: List[str], sources: List[str] = None) -> str:
        '''
        Build a compact prompt: short system instruction (“answer concisely or say unknown”), 
        then Context (bullet list of passages), then the Question. 
        '''
        context = "\n\n".join([f"- {p}" for p in passages])
        src_block = ""
        if self.include_sources and sources:
            src_block = "\n\nSources:\n" + "\n".join(sources)
        # System instruction
        sys_inst = ("You are a precise assistant. Answer the question with a short factual span. "
                    "If unsure, say 'unknown'.")
        # Final prompt
        prompt = (f"<s>[SYSTEM]\n{sys_inst}\n[/SYSTEM]\n"
                  f"[USER]\nContext:\n{context}{src_block}\n\nQuestion: {question}\n[/USER]\n[ASSISTANT]\n")
        # # Alternative
        # prompt = self.tok.apply_chat_template(
        #     [{"role":"system","content":sys_inst},
        #      {"role":"user","content": f"Context:\n{context}\n\nQuestion: {question}"}],
        #      tokenize=False, add_generation_prompt=True)
        return prompt


    def answer(self, question: str, passages: List[str], sources: List[str] = None) -> str:
        '''
        Call .generate() on the LLM with user prompt; return one short string (first line after [ASSISTANT])
        as the system’s answer for the question.
        '''
        # Converts the prompt string to token IDs and moves them to the same device as the model
        prompt = self._format_prompt(question, passages, sources)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.lm.generate(
                **ids,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tok.eos_token_id
            )
        # Decode the generated token IDs back to a string
        text = self.tok.decode(out[0], skip_special_tokens=True)
        # post-process: take the segment after last [ASSISTANT]
        ans = text.split("[ASSISTANT]")[-1].strip()
        return ans.split("\n")[0].strip()


#################################################################################
# Persistence Helpers
#################################################################################
def save_index(outdir: str, texts: List[str], metas: List[Dict], bm25: BM25Index, dense: DenseIndex):
    '''Save texts, metas, bm25, faiss index, and dense meta to outdir as pickle & JSON files.'''
    os.makedirs(outdir, exist_ok=True)
    # texts: raw chunk texts (list of strings)
    with open(os.path.join(outdir, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)
    # metas: metadata per chunk (list of dicts)
    with open(os.path.join(outdir, "metas.pkl"), "wb") as f:
        pickle.dump(metas, f)
    # bm25.tokenized_docs: tokenized corpus for BM25, which is List[List[str]]
    with open(os.path.join(outdir, "bm25.pkl"), "wb") as f:
        pickle.dump({"tokenized_docs": bm25.tokenized_docs}, f)
    # faiss + dense meta: FAISS vector index and minimal metadata to rebuild the dense retriever at query time
    faiss.write_index(dense.index, os.path.join(outdir, "faiss.index"))
    with open(os.path.join(outdir, "dense_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model_name": dense.model_name, "dim": dense.dim, "normalize": dense.normalize}, f)


def load_index(indexdir: str) -> Tuple[List[str], List[Dict], BM25Index, DenseIndex]:
    '''Load texts, metas, bm25, faiss index, and dense meta pickle & JSON files from indexdir.'''
    # texts
    with open(os.path.join(indexdir, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    # metas
    metas_path = os.path.join(indexdir, "metas.pkl")
    if os.path.exists(metas_path):
        with open(metas_path, "rb") as f:
            metas = pickle.load(f)
    else:
        metas = [None] * len(texts)
    # bm25
    with open(os.path.join(indexdir, "bm25.pkl"), "rb") as f:
        bobj = pickle.load(f)
    bm25 = BM25Index(BM25Okapi(bobj["tokenized_docs"]), bobj["tokenized_docs"])
    # faiss + dense meta
    with open(os.path.join(indexdir, "dense_meta.json"), "r", encoding="utf-8") as f:
        dense_meta = json.load(f)
    faiss_index = faiss.read_index(os.path.join(indexdir, "faiss.index"))
    dense = DenseIndex(model_name=dense_meta["model_name"], dim=dense_meta["dim"],
                       index=faiss_index, normalize=dense_meta["normalize"])
    return texts, metas, bm25, dense


#################################################################################
# Command-Line: Index
#################################################################################
def cmd_index(args):
    '''Build BM25 + FAISS indexes from chunks; save to outdir'''
    rows = read_jsonl(args.chunks)
    texts = [get_chunk_text(r) for r in rows]
    metas = [get_chunk_meta(r) for r in rows]
    print(f"Loaded {len(texts)} chunks")

    # BM25
    print("Building BM25...")
    bm25 = BM25Index.build(texts)

    # Dense
    print(f"Building dense index with {args.embedder} ...")
    dense = DenseIndex.build(texts, model_name=args.embedder, normalize=True, batch=args.batch)

    # Save into pickle & JSON files
    save_index(args.outdir, texts, metas, bm25, dense)
    print(f"Saved indexes to {args.outdir}")


#################################################################################
# Command-Line: Answer
#################################################################################
def read_questions(path: str) -> Dict[str, str]:
    '''
    Read a text file with one question per line; return a dict of {line_number: question}. 
    Line counter starts at 1.
    '''
    questions = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                questions[str(i)] = line.strip()
    return questions


def cmd_answer(args):
    '''Run retrieval + reader; write system_outputs/system_output_.json'''
    texts, metas, bm25, dense = load_index(args.indexdir)

    if len(metas) != len(texts):      # alignment check
        metas = (metas + [None] * len(texts))[:len(texts)]

    print(f"Loaded index with {len(texts)} chunks")

    reader = Reader(args.reader, max_new_tokens=args.max_new_tokens, temperature=args.temp, include_sources=args.include_sources)
    questions = read_questions(args.questions)
    outputs = {}

    # Retrieve top-k chunks' indices for each question
    for q_id, q in tqdm(questions.items(), desc="Answering"):
        if args.mode == "sparse":
            idx, _ = bm25.retrieve(q, args.k)
            top_ids = idx.tolist()
        elif args.mode == "dense":
            idx, _ = dense.retrieve(q, args.k)
            top_ids = idx.tolist()
        elif args.mode == "hybrid":
            if args.fusion == "weighted":
                top_ids, _ = hybrid_weighted(args.k, bm25, dense, q, alpha=args.alpha, pool=args.pool)
            else:
                top_ids, _ = hybrid_rrf(args.k, bm25, dense, q, pool=args.pool, K=args.rrfK)
        else:
            raise ValueError("mode must be one of: sparse, dense, hybrid")

        # gather passages for reader
        passages = [texts[i] for i in top_ids]

        # Optionally log sources
        src_lines = []
        for i, idx in enumerate(top_ids):
            m = metas[idx] or {}
            title = (m.get("title") or "").strip()
            url   = (m.get("url")   or "").strip()
            if title or url:
                src_lines.append(f"[{i+1}] {title} — {url}")
        if src_lines:
            print(f"Q{q_id} sources:\n  " + "\n  ".join(src_lines))

        # read and generate
        ans = reader.answer(q, passages)
        outputs[q_id] = ans

    # write output answers as JSON
    outdir = os.path.dirname(args.system_out) or "."
    os.makedirs(outdir, exist_ok=True)
    with open(args.system_out, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.system_out}")


#################################################################################
# Arg Parsing
#################################################################################
def main():
    p = argparse.ArgumentParser(description="RAG (BM25 + FAISS + Hybrid + HF reader)")
    sp = p.add_subparsers()

    # Index command
    p_idx = sp.add_parser("index", help="Build BM25 + FAISS over chunks")
    p_idx.add_argument("--chunks", required=True)
    p_idx.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p_idx.add_argument("--batch", type=int, default=256)
    p_idx.add_argument("--outdir", default="index")
    p_idx.set_defaults(func=cmd_index)

    # Answer command
    p_ans = sp.add_parser("answer", help="Run retrieval + reader and write system_output_.json")
    p_ans.add_argument("--questions", required=True)
    p_ans.add_argument("--indexdir", default="index")
    p_ans.add_argument("--reader", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # choose any HF open-weight model ≤32B
    p_ans.add_argument("--k", type=int, default=8)
    p_ans.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default="hybrid")
    p_ans.add_argument("--fusion", choices=["weighted", "rrf"], default="weighted")
    p_ans.add_argument("--alpha", type=float, default=0.5)  # weight for dense in weighted fusion
    p_ans.add_argument("--pool", type=int, default=50)
    p_ans.add_argument("--rrfK", type=int, default=60)
    p_ans.add_argument("--max_new_tokens", type=int, default=64)
    p_ans.add_argument("--temp", type=float, default=0.0)
    p_ans.add_argument("--system_out", default="system_outputs/system_output_.json")
    p_ans.add_argument("--include_sources", default=False)     # set as False for submission. True for demos
    p_ans.set_defaults(func=cmd_answer)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()

