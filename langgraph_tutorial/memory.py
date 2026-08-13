import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)

class AgentState(TypedDict):
    messages : Annotated[List, add_messages]


def welcome(state: AgentState) -> AgentState:
    """simple node that greets user"""

    cur_message = state['messages']

    response = llm.invoke(cur_message).content

    return {"messages": [("assistant", response)]}

graph = StateGraph(AgentState)

graph.add_node("welcome", welcome)
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)

checkpoint = InMemorySaver()
memory_graph = graph.compile(checkpointer = checkpoint)


config = {'configurable': {'thread_id': 'yogesh'}}

response1 = memory_graph.invoke(
    {"messages": "i am yogi"},
    config
)

response2 = memory_graph.invoke(
    {"messages": "what is my name?"},
    config
)

for message in response2['messages']:
    message.pretty_print()
