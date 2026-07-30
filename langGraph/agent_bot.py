import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
from typing import TypedDict, List

load_dotenv()

class AgentState(TypedDict):
    messages:  List[HumanMessage]

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)
def process(state : AgentState) -> AgentState:

    response =  llm.invoke(state["messages"])
    print(f"AI : {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START , "process")
graph.add_edge("process" , END)
agent = graph.compile()


user_input = input("Chat : ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Chat : ")
