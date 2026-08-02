from sympy.ntheory import continued_fraction
from tenacity import retry_unless_exception_type
import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

document_content = ""

llm = ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)

@tool
def update(content:str) -> str:
   """which tool to update the document content"""

   global document_content
   document_content = content
   return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename =f"{filename}.txt"

    with open(filename, 'w') as file:
        file.write(document_content)
    
    print(f"Document succesfully saved to this file {filename}")
    return f"Document succesfull saved to this file {filename}"


tools = [update,save]

model = llm.bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state['messages']:
        user_input = "I am ready to help you to update the document , what you like to create"
        user_messages = [HumanMessage(content=user_input)]

    else :
        user_input = input("\n what you like to do document? ")
        print(f"\n👤 USER: {user_input}")
        user_messages =  [HumanMessage(content=user_input)]

    all_messages = [system_prompt] +  state['messages'] + user_messages

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc["name"] for tc in response.tool_calls]}")
    
    return {"messages": list(state['messages']) + user_messages + [response]}


def sholud_continue(state: AgentState) -> AgentState:
    """Determine wheather to continue or end the agent"""

    messages = state['messages']

    if not messages:
        return "continue"

    for message in reversed(messages):
        if(isinstance(message, ToolMessage) and
        "saved" in message.content.lower() and
        "document" in message.content.lower()):
            return "end"
        
    return "continue"

def print_message(messages):

    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent",our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent","tools")

graph.add_conditional_edges(
    "tools",
    sholud_continue,
    {
       "continue": "agent",
       "end": END 
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()






    



