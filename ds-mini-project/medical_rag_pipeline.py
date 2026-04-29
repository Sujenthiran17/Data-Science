import os
import requests
import xml.etree.ElementTree as ET
import chromadb
import google.generativeai as genai
from typing import List, Dict

# 1. Configuration & Setup

GEMINI_API_KEY = "AQ.Ab8RN6K_FL4XHvigYfTozXwWMAKPP4JuVZ72JvlKaVODN9rf1g"
genai.configure(api_key=GEMINI_API_KEY)

# Using Gemini for generation
generation_model = genai.GenerativeModel('models/gemini-flash-latest')

# 2. PubMed Retrieval (NCBI E-utilities)

def search_pubmed(query: str, max_results: int = 3) -> List[str]:
    """Search PubMed and return a list of article IDs."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    data = response.json()
    return data["esearchresult"].get("idlist", [])

def fetch_pubmed_abstracts(id_list: List[str]) -> List[Dict[str, str]]:
    """Fetch article titles and abstracts using PubMed IDs."""
    if not id_list:
        return []
    
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    response = requests.get(fetch_url, params=params)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    articles = []
    
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle")
        
        # Abstracts can have multiple parts (Background, Methods, etc.)
        abstract_texts = article.findall(".//AbstractText")
        abstract = " ".join([elem.text for elem in abstract_texts if elem.text])
        
        if title and abstract:
            articles.append({
                "id": pmid,
                "title": title,
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
            
    return articles


# 3. Document Chunking

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap to retain context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


# 4. Embeddings & Vector Store Setup

class MedicalRAG:
    def __init__(self):
        # Initialize local persistent ChromaDB client instance
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
        
        # Reset collection if it already exists
        try:
            self.chroma_client.delete_collection("medical_rag")
        except:
            pass
            
        self.collection = self.chroma_client.create_collection("medical_rag")

    def embed_and_store(self, articles: List[Dict[str, str]]):
        """Process articles into chunks, compute embeddings, and store them."""
        for article in articles:
            full_text = f"Title: {article['title']}\nAbstract: {article['abstract']}"
            chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
            
            for i, chunk in enumerate(chunks):
                # 1. Generate embedding using Gemini
                embedding_response = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=chunk,
                    task_type="retrieval_document"
                )
                embedding = embedding_response['embedding']
                
                # 2. Store in ChromaDB
                doc_id = f"pmid_{article['id']}_chunk_{i}"
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": article['url'], "title": article['title']}]
                )
        print(f"Stored {len(articles)} articles in the vector database.")

    # ==========================================
    # 5. Retrieval & Generation Pipeline
    # ==========================================
    def ask_medical_question(self, query: str) -> str:
        """End-to-end RAG pipeline: Retrieve relevant chunks -> Generate response."""
        # 1. Embed the user query
        query_embedding_response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_response['embedding']
        
        # 2. Retrieve top-k similar chunks from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3  # Top 3 most relevant chunks
        )
        
        retrieved_docs = results['documents'][0]
        metadata = results['metadatas'][0]
        
        if not retrieved_docs:
            return "No relevant medical context could be found to answer this question."
            
        # Compile the retrieved context
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = metadata[i]['source']
            context_parts.append(f"--- Document snippet (Source: {source}) ---\n{doc}")
        context = "\n\n".join(context_parts)
        
        # 3. Generation using prompt engineering (Anti-hallucination + Disclaimer)
        prompt = f"""
You are a highly capable medical research assistant. Use ONLY the provided context from PubMed scientific papers to answer the user's question. 

### CONTEXT:
{context}

### USER QUESTION:
{query}

### INSTRUCTIONS:
1. Answer the question comprehensively but simply, based STRICTLY on the context provided above.
2. If the context does not contain sufficient information to answer the question, state: "I cannot answer this based on the retrieved medical articles." Do NOT guess or hallucinate.
3. Cite the sources provided in the context format (e.g., [Source URL]).
4. Include the following disclaimer at the very end of your response:
   "**Disclaimer:** This information is generated for research purposes based on PubMed abstracts and should not be considered medical advice. Always consult a qualified healthcare provider for medical decisions."
"""
        
        # Call Gemini to generate the structured answer
        response = generation_model.generate_content(prompt)
        return response.text


# Example Workflow Execution

if __name__ == "__main__":
    print('============================================')
    print(' PubMed Medical RAG Pipeline - Initialization')
    print('============================================\n')
    
    # User's query
    user_query = "Latest treatment for diabetes"
    print(f"User Query: '{user_query}'\n")
    
    # Step 1: Search PubMed API
    print("1. Searching PubMed for relevant articles...")
    pmids = search_pubmed(user_query, max_results=3)
    print(f"   Found Article PMIDs: {pmids}")
    
    # Step 2: Fetch Abstracts
    print("\n2. Fetching abstracts and titles...")
    articles = fetch_pubmed_abstracts(pmids)
    for a in articles:
        print(f"   - {a['title'][:60]}...")
        
    # Step 3 & 4: Embed and Store
    print("\n3. Chunking, Embedding, and Indexing... (Using Gemini models)")
    pipeline = MedicalRAG()
    pipeline.embed_and_store(articles)
    
    # Step 5: Querying RAG
    print("\n4. Retrieving context & Generating Answer...\n")
    answer = pipeline.ask_medical_question(user_query)
    
    print('============================================')
    print(' Generated Response')
    print('============================================\n')
    # Use sys.stdout.buffer to print UTF-8 to avoid Windows console encoding errors
    import sys
    sys.stdout.buffer.write(answer.encode('utf-8'))
    print("\n")
