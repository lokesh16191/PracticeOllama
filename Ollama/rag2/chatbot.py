# app.py
import streamlit as st
from io import BytesIO
import pandas as pd
import pdfplumber
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import math
import tempfile

st.set_page_config(page_title="Streamlit → FAISS Embeddings", layout="wide")

# -------- utils: file reading --------
def extract_text_from_pdf(file_bytes: bytes) -> List[Dict]:
    texts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                texts.append({"source": "pdf", "page": i, "text": txt})
    return texts

def extract_text_from_excel(file_bytes: bytes) -> List[Dict]:
    # read all sheets and join textual cells
    buf = BytesIO(file_bytes)
    xl = pd.read_excel(buf, sheet_name=None, engine="openpyxl")
    results = []
    for sheet_name, df in xl.items():
        # convert each row to text join
        for i, row in df.iterrows():
            row_text = " ".join(map(str, row.values.tolist()))
            if str(row_text).strip() and str(row_text).lower() != "nan":
                results.append({"source": "excel", "sheet": sheet_name, "row": int(i+1), "text": str(row_text)})
    return results

def extract_text_from_csv(file_bytes: bytes) -> List[Dict]:
    buf = BytesIO(file_bytes)
    df = pd.read_csv(buf, dtype=str, keep_default_na=False)
    results = []
    for i, row in df.iterrows():
        row_text = " ".join([str(v) for v in row.values.tolist()])
        if row_text.strip():
            results.append({"source": "csv", "row": int(i+1), "text": row_text})
    return results

def extract_text_from_txt(file_bytes: bytes) -> List[Dict]:
    txt = file_bytes.decode("utf-8", errors="ignore")
    blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
    return [{"source": "txt", "block": i+1, "text": b} for i, b in enumerate(blocks, start=1)]

# -------- text chunking with overlap (semantic overlap) --------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Simple sliding window chunker by characters (can be by words if preferred).
    chunk_size and overlap measured in characters by default.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # overlap between windows
        if start < 0:
            start = 0
        if start >= text_len:
            break
    return chunks

# alternative: chunk by words (optional)
def chunk_text_by_words(text: str, chunk_words:int=150, overlap_words:int=30) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks=[]
    i=0
    while i < len(words):
        chunk = words[i:i+chunk_words]
        chunks.append(" ".join(chunk))
        i = i + chunk_words - overlap_words
    return chunks

# -------- build embeddings + faiss index helpers --------
def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    # normalize rows for cosine similarity when using IndexFlatIP
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms==0] = 1e-9
    return emb / norms

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings.astype(np.float32))
    return index

# -------- Pipeline to convert uploaded files to chunks + metadata --------
def files_to_chunks(uploaded_files, chunk_mode="char", chunk_size=800, overlap=200, by_words=False, chunk_words=150, overlap_words=30) -> Tuple[List[str], List[Dict]]:
    """
    Returns (list_of_chunks, list_of_metadatas) where metadata at same index has source info
    """
    all_chunks = []
    metadatas = []
    for up in uploaded_files:
        name = up.name
        data = up.getvalue()
        lower = name.lower()
        if lower.endswith(".pdf"):
            blocks = extract_text_from_pdf(data)
        elif lower.endswith((".xlsx", ".xls")):
            blocks = extract_text_from_excel(data)
        elif lower.endswith(".csv"):
            blocks = extract_text_from_csv(data)
        elif lower.endswith(".txt"):
            blocks = extract_text_from_txt(data)
        else:
            st.warning(f"Skipping unsupported file type: {name}")
            continue

        for block in blocks:
            text = block.get("text", "")
            if by_words:
                chunks = chunk_text_by_words(text, chunk_words, overlap_words)
            else:
                chunks = chunk_text(text, chunk_size, overlap)
            for i, c in enumerate(chunks):
                all_chunks.append(c)
                md = {"source_file": name, "orig_meta": block, "chunk_id": len(all_chunks)-1, "chunk_index_within_block": i}
                metadatas.append(md)
    return all_chunks, metadatas

# -------- Streamlit UI --------
st.title("📄 Streamlit → FAISS (embeddings + semantic overlapping)")
st.markdown("Upload PDF/Excel/CSV/TXT files, create overlapping chunks, embed with sentence-transformers, index in FAISS, and run semantic search.")

uploaded = st.file_uploader("Upload files (multiple allowed)", type=["pdf","xlsx","xls","csv","txt"], accept_multiple_files=True)

with st.sidebar:
    st.header("Indexing settings")
    model_name = st.selectbox("Embedding model (local)", options=[
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "sentence-transformers/all-distilroberta-v1"
    ], index=0)
    chunk_mode = st.radio("Chunk mode", ("characters (fast)", "words (semantic)"))
    if chunk_mode == "characters (fast)":
        use_words = False
        chunk_size = st.number_input("Chunk size (characters)", min_value=200, max_value=4000, value=800, step=100)
        overlap = st.number_input("Overlap (characters)", min_value=0, max_value=2000, value=200, step=50)
        chunk_words = 0; overlap_words = 0
    else:
        use_words = True
        chunk_words = st.number_input("Chunk size (words)", min_value=50, max_value=1000, value=150, step=10)
        overlap_words = st.number_input("Overlap (words)", min_value=0, max_value=500, value=30, step=5)
        chunk_size = 0; overlap = 0

    batch_size = st.number_input("Embedding batch size", min_value=16, max_value=1024, value=64, step=16)
    save_dir = st.text_input("Directory to save index (local)", value="faiss_index_data")
    os.makedirs(save_dir, exist_ok=True)

index_state = st.session_state.get("faiss_index", None)
meta_state = st.session_state.get("metadatas", None)
emb_model = None

if st.button("Create / Rebuild index from uploaded files"):
    if not uploaded:
        st.error("Upload at least one file first.")
    else:
        with st.spinner("Extracting text and creating chunks..."):
            chunks, metadatas = files_to_chunks(uploaded, chunk_mode, chunk_size, overlap, by_words=use_words, chunk_words=chunk_words, overlap_words=overlap_words)
        st.write(f"Total chunks created: {len(chunks)}")
        if len(chunks) == 0:
            st.error("No chunks extracted from uploaded files.")
        else:
            with st.spinner(f"Loading embedding model `{model_name}`..."):
                emb_model = SentenceTransformer(model_name)
            # embed in batches
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                emb = emb_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                embeddings.append(emb)
            embeddings = np.vstack(embeddings)
            embeddings = normalize_embeddings(embeddings)
            # build faiss index
            index = build_faiss_index(embeddings)
            # save index and metadatas
            idx_path = os.path.join(save_dir, "faiss.index")
            faiss.write_index(index, idx_path)
            with open(os.path.join(save_dir, "metadatas.pkl"), "wb") as f:
                pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)
            # store in session state
            st.session_state["faiss_index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["metadatas"] = metadatas
            st.success(f"Index built and saved to `{save_dir}` ({len(chunks)} chunks).")

if st.button("Load index from disk"):
    idx_path = os.path.join(save_dir, "faiss.index")
    md_path = os.path.join(save_dir, "metadatas.pkl")
    if os.path.exists(idx_path) and os.path.exists(md_path):
        index = faiss.read_index(idx_path)
        with open(md_path, "rb") as f:
            data = pickle.load(f)
        st.session_state["faiss_index"] = index
        st.session_state["chunks"] = data["chunks"]
        st.session_state["metadatas"] = data["metadatas"]
        st.success("Index + metadata loaded into session.")
    else:
        st.error("Index files not found in save directory.")

# Search UI
st.markdown("---")
st.header("Search the index")
query = st.text_input("Enter a semantic query")
top_k = st.slider("Top K results", min_value=1, max_value=20, value=5, step=1)

if st.button("Search"):
    if "faiss_index" not in st.session_state:
        st.error("No index in memory. Build or load the index first.")
    elif not query.strip():
        st.error("Enter a query.")
    else:
        # lazy load model for queries if needed
        if emb_model is None:
            emb_model = SentenceTransformer(model_name)
        q_emb = emb_model.encode([query], convert_to_numpy=True)
        q_emb = normalize_embeddings(q_emb)
        index = st.session_state["faiss_index"]
        D, I = index.search(q_emb.astype(np.float32), top_k)
        D = D[0]; I = I[0]
        chunks = st.session_state["chunks"]
        metadatas = st.session_state["metadatas"]

        results = []
        seen_texts = set()
        st.write(f"Top {top_k} results (score = cosine similarity):")
        for score, idx in zip(D, I):
            if idx < 0 or idx >= len(chunks):
                continue
            txt = chunks[idx]
            # optional dedupe: skip near-duplicate chunks to reduce repeated overlapping results shown
            key = txt[:200]
            if key in seen_texts:
                continue
            seen_texts.add(key)
            md = metadatas[idx]
            results.append((float(score), idx, txt, md))

        for score, idx, txt, md in results:
            st.subheader(f"Score: {score:.4f} — chunk #{idx}")
            st.caption(f"Source: {md.get('source_file')} — meta: {md.get('orig_meta')}")
            st.write(txt)

st.markdown("---")
st.info("Notes: Overlapping chunks help capture semantics that span chunk boundaries — good when text has consecutive/connected ideas.")

