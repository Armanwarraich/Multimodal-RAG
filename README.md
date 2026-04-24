# OmniRAG — Fully Offline Multimodal Retrieval-Augmented Generation System

A production-style multimodal RAG system capable of ingesting, retrieving, and answering over:

- Text
- PDFs
- Images
- Audio
- Video

All running locally without external APIs or cloud dependencies.

Built and optimized to run on a consumer NVIDIA RTX 3050 Laptop GPU (4GB VRAM).

---

## Key Highlights

- Fully offline privacy-preserving RAG pipeline
- Multimodal retrieval over 5 data modalities
- Local LLM inference using Phi-3 Mini (GGUF via llama-cpp)
- VRAM-aware sequential model orchestration for low-memory GPUs
- Semantic retrieval using FAISS vector search
- Real-time response streaming via FastAPI + React
- Integrated evaluation framework for retrieval quality and latency

---

## Tech Stack

### Models
- Phi-3-mini-4k-instruct (GGUF)
- all-MiniLM-L6-v2 embeddings
- Qwen2-VL-2B-Instruct (4-bit)
- Whisper Small
- OpenCLIP

### Infrastructure
- Python
- FastAPI
- React + Vite + Tailwind
- FAISS
- PyTorch CUDA
- FFmpeg

---

## Architecture

User Query
→ Modality-specific ingestion
→ Chunking / embeddings
→ FAISS retrieval
→ Context grounding
→ Local LLM generation
→ Streaming grounded response

Optimized using sequential loading/unloading to remain within 4GB VRAM.

---

## Evaluation

Implemented evaluation framework includes:

- Recall@k
- Mean Reciprocal Rank (MRR)
- Faithfulness scoring
- Hallucination-rate proxy
- Query latency benchmarking

Observed latency:

- PDF queries: ~3.3 sec
- Complex text queries: ~8–9 sec
- Consumer GPU: RTX 3050 (4GB)

---

## Sample Capabilities

### PDF RAG
Ask:
- Summarize uploaded report
- Extract key risks
- Compare financial figures

### Image Understanding
Ask:
- Describe uploaded image
- Identify objects/scenes

### Audio / Video
Ask:
- Summarize spoken content
- Query video transcripts + keyframes

---

## Interesting Engineering Challenges Solved

- CUDA setup for 4GB VRAM constraint
- Silent-video handling in video ingestion pipeline
- FFmpeg audio extraction integration
- Whisper + video keyframe multimodal processing
- VRAM cleanup orchestration between model stages
- Dependency conflict resolution (NumPy/OpenCV/PyTorch stack)

---

## Repository Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uvicorn python-api.main:app --reload
````

---

## Future Improvements

* Cross-encoder reranking
* Citation-grounded answer generation
* Hybrid sparse+dense retrieval
* Docker deployment
* API-served model backend

---

## Why This Project

This project was designed to explore how far a multimodal RAG system can be pushed on limited hardware while preserving privacy and grounding responses in retrieved evidence.

It focuses not just on model usage, but on systems engineering under constraints.

```

---


- “4GB VRAM constrained orchestration”
- “Integrated evaluation framework”
- “Fully offline multimodal RAG”
- “Systems engineering under hardware constraints”


---

Built a fully offline multimodal RAG system supporting text, PDF, image, audio and video retrieval, optimized to run on 4GB VRAM using custom sequential model orchestration and evaluated using Recall@k, MRR, faithfulness and latency benchmarks.
