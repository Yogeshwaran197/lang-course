import os
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)


class llm_schema(BaseModel):
    funny_flag : Literal['funny', 'not funny'] = Field(..., description="wheather joke is funny or not funny")
    feedback: str = Field(..., description="feedback to improve the joke further")


llm_with_schema = llm.with_structured_output(llm_schema)


class Agentstate(TypedDict):
    topic: str
    joke: str
    funny_flag: str
    feedback: str
    max_iteration: int


def generator_node(state: Agentstate) -> Agentstate:

    topic = state['topic']

    if state['feedback']:
        result = llm.invoke(f"Please modify this joke {state['joke']} based on the feedback {state['feedback']}, only give joke no instruction")

    else:
        result = llm.invoke(f"generate the joke for following topic {topic}, keep it short and concise manner, just give joke")

    state['joke'] =  result.content

    return state

def evaluator_node(state: Agentstate) -> Agentstate:

    joke = state['joke']

    iteration =  state["max_iteration"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an comedy max amini, evaluate the given joke based on max amini style."),
        ("user",f"Evaluate the following joke: {joke}\nRespond with 'funny' or 'not_funny' and provide feedback if it's not funny.")
    ])

    chain = prompt | llm_with_schema

    response = chain.invoke({"joke": joke})

    state['feedback'] =  response.feedback
    state['funny_flag'] =  response.funny_flag
    state['max_iteration'] = iteration  + 1

    return state


def should_continue(state: Agentstate) -> str:
   
    iteration = state['max_iteration']
    if iteration < 5 and state["funny_flag"] != "not funny":
        return "continue"
    else:
        return "End"




graph =  StateGraph(Agentstate)

graph.add_node("generator", generator_node)
graph.add_node("evaluator", evaluator_node)

graph.add_edge(START, "generator")
graph.add_conditional_edges(
    "generator", 
    should_continue,
    {
        "continue": "evaluator",
        "End": END
    } )

graph.add_edge("evaluator", "generator")

generator_eval = graph.compile()


response =  generator_eval.invoke({
    "topic": "persians",
    "joke": "",
    "funny_flag": "",
    "feedback": "",
    "max_iteration": 0
})


print("="*60)
print("Generator Evaulator WorkFlow")
print("=" * 60)

print(f"\nTopic : {response['topic']}")
print(f"\nJoke : {response['joke']}")
print(f"\nFunny Flag : {response['funny_flag']}")
print(f"\nFeedback : {response['feedback']}")
print(f"\nIteration : {response['max_iteration']}")



