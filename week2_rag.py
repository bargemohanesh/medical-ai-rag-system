import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path

print("="*60)
print("WEEK 2: RAG SYSTEM FOR MEDICAL AI")
print("="*60)

# 1. Load documents
print("\n1. Loading medical knowledge base...")
knowledge_base = []
kb_path = Path("knowledge_base")

for file_path in kb_path.glob("*.txt"):
    with open(file_path, 'r') as f:
        content = f.read()
        # Split into chunks (simple splitting by paragraphs)
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        for i, chunk in enumerate(chunks):
            knowledge_base.append({
                "text": chunk,
                "source": file_path.name,
                "chunk_id": i
            })

print(f"   ✅ Loaded {len(knowledge_base)} text chunks from {len(list(kb_path.glob('*.txt')))} documents")

# 2. Create embeddings and vector database
print("\n2. Creating vector database...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="medical_knowledge")

# Add documents to vector DB
if collection.count() == 0:
    print("   Embedding documents...")
    texts = [doc["text"] for doc in knowledge_base]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        ids=[f"doc_{i}" for i in range(len(texts))],
        metadatas=[{"source": doc["source"], "chunk_id": doc["chunk_id"]} for doc in knowledge_base]
    )
    print(f"   ✅ Indexed {len(texts)} chunks")
else:
    print(f"   ✅ Using existing index with {collection.count()} chunks")

# 3. Load fine-tuned model
print("\n3. Loading your fine-tuned medical model...")
base_path = "/mnt/c/Users/Mohanesh/models/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    dtype=torch.bfloat16,
    device_map="cuda:0"
)
model = PeftModel.from_pretrained(base_model, "./week1_model_final")
print("   ✅ Model loaded")

# 4. RAG Query Function
def rag_query(question, top_k=3):
    """Query with RAG: Retrieve relevant docs and generate answer"""
    
    # Retrieve relevant documents
    query_embedding = embedding_model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    
    # Build context from retrieved documents
    context_parts = []
    sources = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(f"[Document {i+1} - {metadata['source']}]\n{doc}")
        sources.append(metadata['source'])
    
    context = "\n\n".join(context_parts)
    
    # Create prompt with context
    prompt = f"""Based on the following medical information, answer the question accurately.

{context}

Question: {question}

Answer:"""
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    answer = response.split("Answer:")[-1].strip()
    
    return {
        "answer": answer,
        "sources": list(set(sources)),
        "context": context
    }

# 5. Test RAG System
print("\n4. Testing RAG System...")
print("="*60)

test_questions = [
    "What is the first-line treatment for Type 2 diabetes?",
    "What are the target blood pressure values for hypertension?",
    "What medications are used for rheumatoid arthritis?"
]

for q in test_questions:
    print(f"\n❓ Question: {q}")
    result = rag_query(q, top_k=2)
    print(f"\n💡 Answer: {result['answer'][:300]}...")
    print(f"\n📚 Sources: {', '.join(result['sources'])}")
    print("-"*60)

print("\n✅ RAG SYSTEM WORKING!")
print("\nYour medical AI can now:")
print("  ✅ Search relevant medical documents")
print("  ✅ Provide context-aware answers")
print("  ✅ Cite sources")
print("\nSaved RAG system to: ./vector_db/")
