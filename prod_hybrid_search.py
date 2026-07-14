from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

  
class HybridSearchEngine:
    def __init__(self, docs, embedding_model="BAAI/bge-m3", k=3, weights=(0.5, 0.5)):
        self.docs = docs
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.k = k
        self.weights = weights

        self.vectorstore = None
        self.vector_search = None
        self.bm25_search = None
        self.hybrid_search = None

        self._build()  # set everything up right away

    def _build(self):
      self.vectorstore = Chroma.from_documents(
      documents=self.docs,
      embedding=self.embeddings,
      collection_name="hybrid_search"
      )

      self.vector_search = self.vectorstore.as_retriever(
      search_kwargs = {
        "k":3
      }
      )

      print("vector search ready")

      self.bm25_search =  BM25Retriever.from_documents(
      self.docs,   
        k=3 
      )

      print("BM25 search ready")

      self.hybrid_search = EnsembleRetriever(
      retrievers = [self.vector_search,self.bm25_search],
      weights = [0.5, 0.5]
      )

    def _run_one(self, query , name, retriever_obj):
      results =  retriever_obj.invoke(query)
      print( "=" * 60)
      print(f"{name}\n - {query} ")
      for i , doc in enumerate(results[:3]):
        print(f"content preview - {i+1}: {doc.page_content[:50]} | soruce: {doc.metadata['source']}")


    def search(self, query):
        self._run_one(query, "VectorSearch", self.vector_search)
        self._run_one(query, "BM25SEARCH", self.bm25_search)
        self._run_one(query, "HybridSearch", self.hybrid_search)

    def run_queries(self, queries):
        for query in queries:
            self.search(query)



docs = [
    Document(page_content="LangChain is a framework for building applications with LLMs, offering chains, agents, and memory.",
             metadata={"source": "langchain_intro.txt", "topic": "framework", "chunk_id": 1}),
    Document(page_content="ChromaDB is an open-source vector database used to store and query embeddings for semantic search.",
             metadata={"source": "chromadb_intro.txt", "topic": "vectorstore", "chunk_id": 2}),
    Document(page_content="BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency, useful for keyword search.",
             metadata={"source": "bm25_intro.txt", "topic": "sparse_retrieval", "chunk_id": 3}),
    Document(page_content="Hybrid search combines dense vector retrieval with sparse keyword retrieval like BM25 to improve recall and precision.",
             metadata={"source": "hybrid_search_intro.txt", "topic": "hybrid_search", "chunk_id": 4}),
    Document(page_content="HuggingFaceEmbeddings wraps sentence-transformer models like BAAI/bge-m3 to generate dense vector representations of text.",
             metadata={"source": "embeddings_intro.txt", "topic": "embeddings", "chunk_id": 5}),
    Document(page_content="Reranking uses a cross-encoder model to re-score retrieved documents against the query for higher precision than the initial retrieval.",
             metadata={"source": "reranking_intro.txt", "topic": "reranking", "chunk_id": 6}),
    Document(page_content="RAGAS is an evaluation framework for RAG pipelines, measuring metrics like faithfulness, answer relevancy, and context precision.",
             metadata={"source": "ragas_intro.txt", "topic": "evaluation", "chunk_id": 7}),
    Document(page_content="ChatGroq provides fast LLM inference by running open models like llama-3.3-70b-versatile on custom LPU hardware.",
             metadata={"source": "groq_intro.txt", "topic": "llm_inference", "chunk_id": 8}),
    Document(page_content="RecursiveCharacterTextSplitter splits long documents into smaller overlapping chunks while trying to preserve semantic boundaries like paragraphs and sentences.",
             metadata={"source": "text_splitter_intro.txt", "topic": "chunking", "chunk_id": 9}),
    Document(page_content="FastAPI is a modern Python web framework used to build APIs quickly, often paired with PGVector for production RAG deployments.",
             metadata={"source": "fastapi_pgvector_intro.txt", "topic": "deployment", "chunk_id": 10}),
]

queries = [
    "what is python?",
    "what is langchain?",
    "what is langGraph?",
    "what is hybrid search?"
]

if __name__ == "__main__":
    engine = HybridSearchEngine(docs)
    engine.run_queries(queries)

