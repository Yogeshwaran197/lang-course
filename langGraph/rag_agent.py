from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)# I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

# Our Embedding Model - has to also be compatible with the LLM
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3",
)


pdf_path = "langGraph\Stock_Market_Performance_2024.pdf"


# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

persist_directory = r"C:\Users\JANARTHAN\lang-course\langGraph"
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> AgentState:
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tool_map = {tc.name: tc for tc in tools}


def llm_agent(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""

    messages = list(state['messages'])
    messages =[SystemMessage(content=system_prompt)] + messages
    messages = llm.invoke(messages)

    return {"messages": [messages]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    
    for t in tool_calls:
        print(f"Tool calls {t['name']} tool with query {t['args'].get('query','no query provided')}")

        results = []
        if t['name'] not in tool_map:
            print(f"Tool name called {t['name']} does not exist")
            result = "Tool not exist or not available right now"
        else:
            result = tool_map[t['name']].invoke(t['args'].get('query',''))
            print(f"Result length: {len(str(result))}")
    
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    
    print(f"Tool excuted succesfully")
    return {"messages": results}


graph = StateGraph(AgentState)

graph.add_node("llm", llm_agent)
graph.add_node("retriever_agent", take_action)

graph.set_entry_point('llm')

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True : "retriever_agent", False : END
    }
)

graph.add_edge("retriever_agent", "llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    running_agent()

    