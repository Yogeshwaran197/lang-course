import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from typing import TypedDict, Literal
from pydantic import BaseModel, Field


load_dotenv()

llm =  ChatGroq(
    model ="llama-3.3-70b-versatile",
    api_key= os.environ["GROQ_API_KEY"]
)


class llm_schema(BaseModel):
    category: Literal['insta','twitter', 'linkedin'] = Field(..., description="Category of post to generate")
    topic: str = Field(..., description="topic of post to generate")


llm_with_schema = llm.with_structured_output(llm_schema)

class AgentState(TypedDict):
   input : str
   topic: str
   post : str
   category :str



def decider_node(state: AgentState) -> AgentState:

    input = state['input']
    response = llm_with_schema.invoke(input)

    category = response.category
    topic = response.topic

    state['topic'] = topic
    state["category"] = category 

    return state


def insta(state: AgentState) -> AgentState:

    topic = state['topic']
    post = llm.invoke(f"Write an instagram post about {topic}, keep your tone casual and entertaining").content

    state['post'] = post

    return {'post': post}

def twitter(state: AgentState) -> AgentState:

    topic = state['topic']
    post = llm.invoke(f"Write an twitter post about {topic}, keep your tone quick").content

    state['post'] = post

    return {'post': post}

def linkedin(state: AgentState) -> AgentState:

    topic = state['topic']
    post = llm.invoke(f"Write an linkedin post about {topic}, keep your proffessional and impresive ").content

    state['post'] = post

    return {'post': post}


def should_continue(state: AgentState) -> str:

    category = state['category']

    if category == "insta":
        return "insta_post"
    elif category == "twitter":
        return "twitter_post"
    elif category == "linkedin":
        return "linkedin_post"
    else:
        raise ValueError("Invalid Category")


graph = StateGraph(AgentState)


graph.add_node("decider", decider_node)
graph.add_node("insta_post", insta)
graph.add_node("twitter_post", twitter)
graph.add_node("linkedin_post", linkedin)

graph.add_edge(START, "decider")
graph.add_conditional_edges(
    "decider",
    should_continue,
    {
        "insta_post": "insta_post",
        "twitter_post": "twitter_post",
        "linkedin_post": "linkedin_post"
    }

)


graph.add_edge("insta_post", END)
graph.add_edge("twitter_post", END)
graph.add_edge("linkedin_post", END)


route_graph = graph.compile()

response = route_graph.invoke({
    "input": "write an instagram post about Artificial Intelligence",
    "topic": "",
    "post": "",
    "category": ""

})

print("="*60)
print(f"Route WorkFlow LangGraph")
print("="*60)

print(f"\nCategory: {response['category']}")
print(f"Topic: {response['topic']}")

print(f"\nPost : {response['post']}\n")

print(route_graph.get_graph().draw_ascii())



