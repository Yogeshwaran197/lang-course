from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_query import  MultiQueryRetriever
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import tempfile



def demo_parent_document_retriever():
  """ Document retriever for both  parent and child spillters"""

  print("=" * 60)
  print("Parent Document Retriever")
  print("=" * 60)

  long_doc = Document(
        page_content="""
# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

Several frameworks exist for building AI agents:

LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
        """,
        metadata={"source": "ai_agents_guide.md"},
    )
  

  parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
  child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)


  vectorstore  =  Chroma(
    collection_name="parent_child_demo",
    embedding_function= embedding_model,
  )

  store = InMemoryStore()

  retriever = ParentDocumentRetriever(
    vectorstore= vectorstore,
    docstore = store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
  )

  retriever.add_documents([long_doc])


  query = " what is langchain?"
  print(f"\nQuery: {query}")

  child_docs = vectorstore.similarity_search(query, k=1)
  print("----Child chunk for precise search")
  print(f"\nLength : {len(child_docs[0].page_content)} chunks")
  print(f"\nContent: {child_docs[0].page_content}")

  print("=" *60)
  
  parent_docs = retriever.invoke(query)
  print("\n---Parent chunk for context")
  print(f"\nLength : {len(parent_docs[0].page_content)} chunks")
  print(f"\nContent: {parent_docs[0].page_content[:200]}")


TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

def create_vectorstore():
  """simple vectorstore  for demos"""

  return Chroma.from_documents(
    documents=TECH_DOCS,
    embedding=embedding_model
  )

def contextual_compressor():
  """"Contextual compressor retriever for revelent data"""

  print("="*60)
  print("Contextual Compression Retriever")
  print("="*60)

  vectorstore = create_vectorstore()
  llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key = os.environ["GROQ_API_KEY"]
    )
  
  compressor = LLMChainExtractor.from_llm(llm)

  compressor_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = vectorstore.as_retriever(
      search_kwargs = {
        "k": 2
      }
    )
  )

  query = "What is LangGraph's persistence feature?"
  print(f"\nquery : {query}")

  base_docs = vectorstore.as_retriever(
    search_kwargs={
      "k" : 2
    }).invoke(query)

  print("\n------Without Compressure-----")
  for doc in base_docs:
    print(f"length: {len(doc.page_content)} chars")
    print(f"content: {doc.page_content[:150]}")

  compressured_docs = compressor_retriever.invoke(query)
  print("\n------With Compressure-----")
  for doc in compressured_docs:
    print(f"length: {len(doc.page_content)} chars")
    print(f"content: {doc.page_content}")


def demo_multi_query_retriever():
    """Multi-Query Retriever generates multiple query perspectives."""

    print("=" * 60)
    print("MULTI-QUERY RETRIEVER")
    print("Generates multiple perspectives on your question")
    print("=" * 60)

    vectorstore = create_vectorstore()

    multi_retriever = MultiQueryRetriever.from_llm(
      retriever= vectorstore.as_retriever(
        search_kwargs = {
          "k":1
        }
      ),
      llm = llm
    )

    query = "What tools can I use to build AI applications?"

    docs = multi_retriever.invoke(query)

    print(f"Retrieved {len(docs)} unique documents:")
    for i, doc in enumerate(docs):
        print(
            f"\n{i+1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}..."
        )

  
if __name__ == "__main__":
  #demo_parent_document_retriever()
  #contextual_compressor()
  demo_multi_query_retriever()




