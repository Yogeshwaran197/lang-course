from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.storage import InMemoryStore
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")

llm = init_chat_model(model="llama-3.3-70b-versatile",model_provider="groq", temperature=0.2)

def create_kb():
  """Create vector database for knowledge base"""
  
  doc = Document(page_content=KNOWLEDGE_BASE , metadata = {"source":"langchain knowledge base.md"})
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 50)
  split_docs =  splitter.split_documents([doc])

  vectorstore = Chroma.from_documents(
    split_docs,
    embedding=embedding_model,
    persist_directory=tempfile.mkdtemp(),

  )

  return vectorstore

def basic_rag_demo():
  vectorstore = create_kb()
  retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
      "k":2
   }
  )

  prompt = ChatPromptTemplate.from_template(
    """Answer the Question only using {context}.
    
    Question: {question}

    Answer: 

    Be concise and  make sure that answer in well explained and if you don't know answer just say I don't know"""
  )

  def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
  

  rag_chain = {"context":retriever | format_docs, "question":RunnablePassthrough()} | prompt | llm | StrOutputParser()

  questions = [
    "what is langchain?",
    "what is langGraph?",
    "what is agents?"
  ]

  print("Basic Rag Demo:\n")
  for q in questions:
    answer = rag_chain.invoke(q)

    print(f"Question: {q}")
    print(f"Answer: {answer}\n")



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



  
if __name__ == "__main__":
  #basic_rag_demo()
  demo_parent_document_retriever()




