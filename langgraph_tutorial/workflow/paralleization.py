import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from typing import TypedDict, List


load_dotenv()

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)


class AgentState(TypedDict):
    topic: str
    insta: str
    twitter: str
    linkedin: str


def insta(state: AgentState) -> AgentState:

    topic = state['topic']
    post = llm.invoke(f"Write an instagram post about {topic}, keep your tone casual and entertaining").content

    state['insta'] = post

    return {'insta': post}

def twitter(state: AgentState) -> AgentState:

    topic = state['topic']
    post = llm.invoke(f"Write an twitter post about {topic}, keep your tone quick").content

    state['twitter'] = post

    return {'twitter': post}

def linkedin(state: AgentState) -> AgentState:

    topic = state['topic']
    post = llm.invoke(f"Write an linkedin post about {topic}, keep your proffessional and impresive ").content

    state['linkedin'] = post

    return {'linkedin': post}


graph = StateGraph(AgentState)

graph.add_node("insta_post", insta)
graph.add_node("twitter_post", twitter)
graph.add_node("linkedin_post", linkedin)

graph.add_edge(START, "insta_post")
graph.add_edge(START, "twitter_post")
graph.add_edge(START, "linkedin_post")

graph.add_edge("insta_post", END)
graph.add_edge("twitter_post", END)
graph.add_edge("linkedin_post", END)


parallel_graph = graph.compile()

response = parallel_graph.invoke({
    "topic":"Artificial Intelligence",
    "insta":"",
    "twitter":"",
    "linkedin":""
})

print("="*60)
print(f"\nInsta Post: {response['insta']}")
print("="*60)

print("="*60)
print(f"\nTwitter Post: {response['twitter']}")
print("="*60)

print("="*60)
print(f"\nLinkedin Post: {response['linkedin']}")
print("="*60)

