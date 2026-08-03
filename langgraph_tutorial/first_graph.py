import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END


load_dotenv()

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)

class AgentState(TypedDict):
    name : str
    messages : str


def welcome(state: AgentState) -> AgentState:
    """simple node that greets user"""

    cur_name = state['name']
    cur_message = state['messages']

    response = llm.invoke(f"my name  is {cur_name},{cur_message}").content

    state['messages'] = f"Your message was {cur_message}. here my response {response}"
    return state

graph = StateGraph(AgentState)

graph.add_node("welcome", welcome)
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)
app = graph.compile()


response = app.invoke({"name": "yogi","messages": "How are you?"})
print(response['messages'])


