import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Dict , List

load_dotenv()

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)


@tool
def duckduckgosearch(query: str):
    """this tools is for fetching realtime infoformation from DuckDuckGoSearch"""
    ducksearch = DuckDuckGoSearchRun()
    result = ducksearch.invoke(query)
    return result

@tool
def Arxiv(query: str):
    """this tools is for fetching realtime infoformation from Arxiv"""
    arxiv = ArxivQueryRun(ArxivAPIWrapper())
    result = arxiv.invoke(query)
    return result

@tool
def wikipedia(query: str):
    """this tools is for fetching realtime infoformation from wikipedia"""
    wiki = WikipediaQueryRun(WikipediaAPIWrapper())
    result = wiki.invoke(query)
    return result


@tool
def personal_info(name: str):
    """Use this tool to get personal information about Alice, Bob, or Charlie."""

    info = {
        "Alice": "Alice is a software engineer with 5 years of experience in AI.",
        "Bob": "Bob is a data scientist who loves working with large datasets.",
        "Charlie": "Charlie is a product manager with a background in tech startups."
    }

    return info.get(name, "No information is abot this name")


tools = [duckduckgosearch, Arxiv, wikipedia, personal_info]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages : List


def llm_node(state: AgentState) -> AgentState:
    """this node is for llm generation"""

    messages = state['messages']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools to answer questions."),
        ("human", "{input}")
    ])

    chain = prompt | llm_with_tools

    response = chain.invoke({"input": messages})

    state['messages'] =  messages + [response]

    return state


def tool_node(state: AgentState) -> AgentState:
    """this is for llm to use tools for unknown answers."""

    messages = state['messages']
    tool_map = {t.name: t for t in tools}

    tool_results = []

    for tool_call in messages[-1].tool_calls:
        tool = tool_map[tool_call['name']]
        observation = tool.invoke(tool_call['args'])

        tool_results.append(ToolMessage(content=str(observation), tool_call_id = tool_call['id']))

    state['messages'] =  messages + tool_results
    return state 


def should_continue(state: AgentState) -> str:
    """decides should continue loop or not"""

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
       "continue": "tool_node",
       "end": END
    }
)

graph.add_edge("tool_node", "llm_node")

ReactAgent = graph.compile()

result = ReactAgent.invoke({'messages' : [HumanMessage(content="What is latest news about ai in 08/08/2026?")]})
print(result)














