import os 
from dotenv import load_dotenv
from typing import Sequence, Annotated , TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from  langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)

@tool
def add(a:int, b:int):
    "Tool that addes two numbers together"
    return a + b

@tool
def sub(a:int, b:int):
    """tool that subs two numberr"""
    return a - b

@tool
def multi(a:int, b:int):
    """tool that that multipiles two number"""
    return a * b

@tool
def div(a:int, b:int):
    """tool that that divides two number"""
    return a / b


tools = [add, sub, multi, div]

model = llm.bind_tools(tools)

def model_call(state: AgentState) -> AgentState:

    systemMessage = SystemMessage(
        content = "You're my helpful AI assistant,  please answer my query to the best of your ability. "
    )
    response = model.invoke([systemMessage] + state['messages'])

    return {"messages": response}

def should_continue(state: AgentState) -> AgentState:

    messages = state['messages']
    last_message =  messages[-1]
    
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph =  StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("our_agent", "tools")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))











