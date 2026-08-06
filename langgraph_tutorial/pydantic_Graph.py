import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from  langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field


load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key= os.environ["GROQ_API_KEY"]
)

class AgentState(BaseModel):

    topic: str = Field(..., description="this is topic")
    post: str = Field(..., description="this is post")
    curated: str = Field(..., description="this is curated post")


def post(state: AgentState) -> AgentState:
    """this is node to generate post for topic"""

    state = state.model_dump()
    topic = state['topic']

    response = llm.invoke(f"write the linkedin post about this {topic}").content
    state['post'] = response

    return state

def curated_post(state: AgentState) -> AgentState:
    """this node is to create curated post"""

    state = state.model_dump()
    post = state['post']

    curated_response  = llm.invoke(f"write curated linkedin post for this {post} in form of GenZ tone").content
    state['curated'] = curated_response

    return state


graph = StateGraph(AgentState)

graph.add_node("create_post", post)
graph.add_node("curated_post", curated_post)

graph.add_edge(START, "create_post")
graph.add_edge("create_post", "curated_post")
graph.add_edge("curated_post", END)

pydantic_graph = graph.compile()

response = pydantic_graph.invoke({
    "topic":"The biggest mistake you made this year and what it taught you.",
    "post":"",
    "curated":""
})

print(f"\nTOPIC : {response['topic']}\n")
print(f"\nPOST : {response['post']}\n")
print(f"\nCURATED POST : {response['curated']}")
