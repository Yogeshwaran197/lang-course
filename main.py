from langchain_core import version as langchain_version
from langgraph import version as lg
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

print(f"Core version: {langchain_version}")
print(f"LangGraph version: {lg}")


def main():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.6)
    response = llm.invoke("say, Set up completed in one word")
    print(f"response from groq: {response}")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.8)
    response_gemini = llm.invoke("say, Set up completed in one word")
    print(f"response from gemini: {response_gemini}")

    print("Setup Completed")


if __name__ == "__main__":
    main()
