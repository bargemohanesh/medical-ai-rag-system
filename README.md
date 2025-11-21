
#  Medical AI RAG System

Production-grade Medical AI Assistant with Fine-tuned Llama 3.1 8B + RAG (Retrieval Augmented Generation)

##  Overview

A specialized medical AI system that combines:
- **Fine-tuned Llama 3.1 8B** on 10,000 medical questions
- **RAG system** for retrieval from medical knowledge base
- **Source citations** for medical accuracy

##  Features

-  Fine-tuned on MedMCQA dataset (10K medical questions)
-  QLoRA 4-bit quantization for efficient training
-  RAG system with ChromaDB vector database
-  Medical document retrieval with citations
-  Runs on RTX 5080 (16GB VRAM)

##  Quick Start
```bash
# Clone repository
git clone https://github.com/bargemohanesh/medical-ai-rag-system.git
cd medical-ai-rag-system

# Install dependencies
pip install -r requirements.txt

# Run RAG system
python week2_rag.py
```

##  Results

### Week 1: Fine-tuning
- **Training time:** 4.2 hours
- **Dataset:** 10,000 medical questions
- **Final loss:** 1.39 (started at 3.67)
- **Final accuracy:** 68.6% (started at 46.9%)
- **Model size:** 53MB (LoRA adapters only)

### Week 2: RAG System
- **Vector database:** 20 medical document chunks
- **Embedding model:** all-MiniLM-L6-v2
- **Retrieval accuracy:** Provides relevant sources with citations

##  Tech Stack

- **Model:** Meta Llama 3.1 8B Instruct
- **Training:** QLoRA (4-bit), PyTorch 2.10, CUDA 12.8
- **RAG:** ChromaDB, Sentence Transformers
- **Hardware:** NVIDIA RTX 5080 (16GB VRAM)

##  Project Structure
```
medical-ai-rag-system/
├── week1_finetune.py      # Fine-tuning script
├── week2_rag.py            # RAG system
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

##  Roadmap

- [x] Week 1: Fine-tune Llama 3.1 8B on medical data
- [x] Week 2: Build RAG system with citations
- [ ] Week 3: Build Gradio web interface
- [ ] Week 4-5: Production stack (Next.js + FastAPI)
- [ ] Week 6-7: Add authentication & analytics
- [ ] Week 8: Deploy to cloud

##  License

MIT

##  Author

**Mohanesh**
- LinkedIn: https://www.linkedin.com/in/mohanesh-barge/
- GitHub: [@bargemohanesh](https://github.com/bargemohanesh)

---

⭐ Star this repo if you find it useful!
