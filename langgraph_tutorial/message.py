from langchain_groq import ChatGroq
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from operator import add
from dotenv import load_dotenv
import os

load_dotenv()

class MessageState(TypedDict):
    message_mannual: List
    message_auto: Annotated[List, add]

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)


def message_1(state: MessageState) -> MessageState:

    message_mannual = state['message_mannual'] 
    response = llm.invoke(message_mannual).content
    response_ai  = AIMessage(content=response)
    state['message_mannual'] = message_mannual + [response_ai]

    message_auto = state['message_auto']
    response_auto = llm.invoke(message_auto).content
    response_auto_ai = AIMessage(content=response_auto)
    state['message_auto'] = [response_auto_ai]

    return state

def message_2(state: MessageState) -> MessageState:

    message_mannual = state['message_mannual']
    response = llm.invoke(message_mannual).content
    response_ai = AIMessage(content=response)
    state['message_mannual'] = message_mannual + [response_ai]

    message_auto = state['message_auto']
    response_auto = llm.invoke(message_auto).content
    response_auto_ai = AIMessage(content=response_auto)
    state['message_auto'] = [response_auto_ai]

    return state


graph = StateGraph(MessageState)

graph.add_node("message_1", message_1)
graph.add_node("message_2", message_2)

graph.add_edge(START, "message_1")
graph.add_edge("message_1", "message_2")
graph.add_edge("message_2", END)

message_graph = graph.compile()

response = message_graph.invoke({
    "message_mannual": [HumanMessage(content="hello")],
    "message_auto": [HumanMessage(content="hello")]
    })

print(response)