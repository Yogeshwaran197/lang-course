"""
LangSmith Setup and Observability
Production monitoring for LangChain/LangGraph

"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree


load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

@traceable(name="Basic_Chaining")
def basic_chain():

  llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
    )


  prompt = ChatPromptTemplate.from_template(
    """ Explain this {topic} in one sentence"""
  )

  chain = prompt | llm | StrOutputParser()

  print("Basic Tracing Demo\n")
  print("Running langchain with langsmith tracing....")

  result = chain.invoke({"topic":"langchain"})

  print(f"Result:\n {result}")
  return result 
  print("Visit Langsmith DashBoard for further details")

@traceable(name="name_run_ones", tags=["production", "summarization"])
def named_runs():
  
  llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
    )


  prompt = ChatPromptTemplate.from_template(
    """ Summarize {text}"""
  )

  chain = prompt | llm | StrOutputParser()

  print("\nNamed Runs Demo:\n")
  result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )
  print(f"Result:\n {result}")
  print("Run tagged with 'production', 'summarization'")
  return result 
  

@traceable(name="Trace with metadata", tags=["metadata", "filtering"])
def trace_with_metadata(user_id):
  llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
    )

    # Metadata is automatically captured
  result = llm.invoke(f"Hello from user {user_id}")

  return result.content


if __name__ == "__main__":
  basic_chain()
  named_runs()
  trace_with_metadata(user_id = "yogi")





  

  

