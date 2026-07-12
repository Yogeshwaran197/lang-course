from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

embedding  = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def chroma_basics():
  vectorstore = Chroma.from_documents(
    documents=SAMPLE_DOCS, embedding= embedding
  )

  print(f"vector store created, {len(vectorstore.get()['ids'])} documents persisted")

  query = "what is langchain?"
  result = vectorstore.similarity_search(query,k=3
  )
    
  for i , doc in enumerate(result):
    print(f"Result-{i+1} : {doc.page_content} | {doc.metadata['source']}")

def similarity_score():
  with tempfile.TemporaryDirectory() as tmpdir:
    vectorstore = Chroma.from_documents(
      documents=SAMPLE_DOCS, embedding=embedding
    )
    
    query = "what is langGraph?"
    results = vectorstore.similarity_search_with_score(query, k = 2)
    for i, (doc, score) in enumerate(results):
      final_score = 1 / (1 + score)  # Convert distance to similarity
      print(
        f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
      )


sample_texts = [
        "Python is a versatile programming language used in web development, "
        "data science, machine learning, and automation. It has a simple syntax "
        "that makes it easy to learn and read.",
        "JavaScript is the language of the web. It runs in browsers and on "
        "servers with Node.js. Modern frameworks like React and Vue make "
        "building web applications efficient.",
        "Rust is a systems programming language focused on safety and "
        "performance. It prevents common bugs like null pointer dereferences "
        "and data races at compile time.",
    ]

def retriever(text:list[str], chunk_size:int, overlap:int):
  docs = [Document(page_content=d) for d in text]
  splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = overlap)
  split_docs = splitter.split_documents(docs)

  vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding
  )

  retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
      "k":3
    }
  )

  result = retriever.invoke("what is python?")
  for i, doc in enumerate(result):
   print(f"Result {i+1}: {doc.page_content}")



if __name__ == "__main__":
  #chroma_basics()
  #similarity_score()
  retriever(sample_texts, 300 , 10)