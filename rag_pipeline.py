from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_query import  MultiQueryRetriever
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
import os
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

def demo_rag_with_sources():

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below. Include which sources you used.

Context:
{context}

Question: {question}

Answer (include sources):"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")




if __name__ == "__main__":
  #basic_rag_dmeo()
  demo_rag_with_sources()