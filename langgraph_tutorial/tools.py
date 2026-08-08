import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

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


message = [HumanMessage(content="What is latest news about ai in 08/08/2026?")]
ai_message = llm_with_tools.invoke(message)
print(f"\n Tool_calls: {ai_message.tool_calls}\n")

tool_map = {t.name:t for t in tools}
print(f"\n{tool_map}\n")









