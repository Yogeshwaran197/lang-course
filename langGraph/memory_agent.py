import os 
from typing import TypedDict, List, Union
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def proccess(state: AgentState) -> AgentState:

    response = llm.invoke(state["messages"])
    state['messages'].append(AIMessage(content=response.content))
    print(f"AI : {response.content}")

    return state


graph =   StateGraph(AgentState)
graph.add_node("proccess", proccess)
graph.add_edge(START, "proccess")
graph.add_edge("proccess", END)
agent = graph.compile()


conversation_history = []

user_input  = input("Enter : ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages" : conversation_history})
    conversation_history = result['messages']
    user_input =  input("Enter : ")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")
