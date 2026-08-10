import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage, ToolMessage
from typing import TypedDict, List
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.prompts import MessagesPlaceholder
import requests

load_dotenv()

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)


@tool
def get_weather(city: str) -> str:
    """This is to get  the weather for city"""
    # Step 1: get lat/lon
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    ).json()
    
    if "results" not in geo:
        return f"Couldn't find location: {city}"
    
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    
    # Step 2: get weather
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True}
    ).json()
    
    temp = weather["current_weather"]["temperature"]
    windspeed = weather["current_weather"]["windspeed"]
    
    return f"{city}: {temp}°C, wind {windspeed} km/h"


@tool
def duckSearch(query: str):
    """This tool is for fetching the realtime data from web browser which is DuckDuckGoSearch"""

    search = DuckDuckGoSearchRun()
    result  = search.invoke(query)
    return result

@tool
def arxivSearch(query: str):
    """this tool is for fetching realtime data from web browser which ArxivSearch"""

    arxiv = ArxivQueryRun(ArxivAPIWrapper())
    result = arxiv.invoke(query)

    return result


tools = [get_weather, duckSearch, arxivSearch]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages : List


def llm_node(state: AgentState) -> AgentState:

    messages = state['messages']

    prompt = ChatPromptTemplate.from_messages([
        ("system","You're Helpful Assistant , you use tools to answer if you did'nt know the answer"),
        MessagesPlaceholder("input"),
    ])

    chain = prompt | llm_with_tools

    result = chain.invoke({"input":messages})

    state["messages"] =  messages + [result]

    return state




def tool_node(state: AgentState) -> AgentState:

    messages = state['messages']
    tool_map = {t.name : t for t in tools}

    tool_results = []
    for tool_call in messages[-1].tool_calls:

        tool = tool_map[tool_call['name']]
        observation = tool.invoke(tool_call['args'])
        
        tool_results.append(ToolMessage(content=observation, tool_call_id = tool_call['id']))
    
    state['messages'] =  messages + tool_results

    return state


def should_continue(state: AgentState) -> AgentState:
    """decided wheather to end or not"""

    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return "continue"
    else:
        return "end"


graph = StateGraph(AgentState)

graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_node")

graph.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        "continue" : "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "llm_node")

reactAgent = graph.compile()


response = reactAgent.invoke({"messages": [HumanMessage(content="""
            what is today news on iran vs usa war?
            also current weather of iran? 
            and lastly give tools name that you used"""
)]})
print(response['messages'][-1].content)



        








