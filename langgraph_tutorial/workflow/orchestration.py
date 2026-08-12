from ddgs import results
from asyncio import tasks
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, List
from pydantic import BaseModel , Field
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

llm =  ChatGroq(
    model ="llama-3.3-70b-versatile",
    api_key= os.environ["GROQ_API_KEY"]
)

class llm_schema(BaseModel):
    task:  List[str] = Field(..., description="List of tasks that to be performer by worked node")

llm_with_schema = llm.with_structured_output(llm_schema)


class AgentState(TypedDict):
    task: list[str]
    query: str
    results: list[str]
    summary: str

def orchestrator(state: AgentState) -> AgentState:

    user_queries = state['query']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an orchestrator, your job is break down the user queries into list of tasks that should be performed by worker."),
        ("user",f"user query : {user_queries} please generate prompt for each task for worker to complete. and return an tasks in list format")
    ])

    chain = prompt | llm_with_schema

    response = chain.invoke({'query' : user_queries})

    state["task"] = response.task

    return state


def excecute(query: str) :

    response  = llm.invoke(f"please excecute this task {query}")

    return response.content


def worker_node(state:AgentState) -> AgentState:

    tasks = state['task']
    results = []

    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        results_furtures = ex.map(excecute, tasks)
        for result in results_furtures:
            results.append(result)
    
    state['results'] = results

    return state

def collector_node(state:AgentState) -> AgentState:

    results = state['results'] 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an summarizer, your job is to summarize the results from worker"),
        ("user", f"here the results from worker {results}, please summarize this in concise manner")
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
graph.add_edge("orchestrator", "worker")
graph.add_edge("worker", "collector")
graph.add_edge("collector", END)

orchestrator_graph = graph.compile()


response =  orchestrator_graph.invoke({
    "query": "Get last recent stock from stock market and anaylse stock loss and profits. then give well structured report. then also give details of stock, owner of stock , what business they run",
    "task": [],
    "summary": "",
    "results": []
})

print("="*60)
print("Orchestration Workflow")
print("="*60)

print(f"\nQuery : {response['query']}")
print(f"\nTasks : {response['task']}")
print(f"\nResults : {response['results'][0]}")
print(f"\nSummary : {response['summary']}")