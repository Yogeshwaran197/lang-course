import os
import operator
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

class llm_schema(BaseModel):
    task: List[str] = Field(..., description="list of tasks that will performed by worker")

llm_with_schema = llm.with_structured_output(llm_schema)


class AgentState(TypedDict):
    task : List[str]
    query: str
    summary: str
    results: Annotated[List[str], operator.add]

def orchestrator(state:AgentState) -> AgentState:

    query = state['query']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "you're an orchestrator , your job is to break down  queries into list of tasks"),
        ("user",f"user query: {query}, please generate prompt for eachh task for worker to complete and return in format of list")
    ])

    chain = prompt | llm_with_schema

    response =  chain.invoke({"query": query})
    
    state['task'] = response.task

    return state

def execute(query):

    response = llm.invoke(f"plese excecute this task : {query}")

    return response.content


def worker_node(state: AgentState) -> dict:

    tasks = state['task']
    result = execute(tasks)

    return {'results': [result]}

def assign_work(state: AgentState) -> AgentState:
    return [Send("worker", {"task" : t }) for t in state['task']]


def collector_node(state:AgentState) -> AgentState:

    results = state['results'] 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an summarizer, your job is to summarize the results from worker"),
        ("user", "here the results from worker {results}, please summarize this in concise manner")
    ])


    chain =  prompt | llm 

    summary =  chain.invoke({'results': results})

    state['summary'] = summary.content

    return state


graph = StateGraph(AgentState)

graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker_node)
graph.add_node("collector", collector_node)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", assign_work, ['worker'])
graph.add_edge("worker", "collector")
graph.add_edge("collector",  END )

orchestrator_graph = graph.compile()

response = orchestrator_graph.invoke({
    "query": "Get last recent stock from stock market and anaylse stock loss and profits. then give well structured report. then also give details of stock, owner of stock , what business they run",
    "task": [],
    "summary": "",
    "results": []
})
print("=" * 60)
print("Orchestration Workflow")
print("=" * 60)
print(f"\nQuery : {response['query']}")
print(f"\nTasks : {response['task']}")
print(f"\nResults : {response['results'][0]}")
print(f"\nSummary : {response['summary']}") 