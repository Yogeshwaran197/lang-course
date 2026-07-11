import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import (TextLoader,PyPDFLoader,WebBaseLoader)

def load_text_file():

  with tempfile.NamedTemporaryFile(delete=False , suffix=".txt") as temp_file:
    temp_file.write(b"hello i am yogeshwaran currently learning GenAI")
    temp_file_path = temp_file.name

  try:
    loader = TextLoader(temp_file_path) 
    documents =loader.load()

    print(f"Totall {len(documents)} loaded")
    print(f"Content Preview: {documents[0].page_content[:100]}")
    print(f"Metadata: {documents[0].metadata}")
  finally:
      os.unlink(temp_file_path)

def pdf_loader(path: str):
  loader = PyPDFLoader(path)
  documents = loader.load()
  print(f"Totally {len(documents)} loaded")
  for i , doc in enumerate(documents):
    print(f"Content preview of {i+1} document : {doc.page_content[:100]}")
    print(f"Metadat: {doc.metadata}")


def web_loader(url: str):
  loader = WebBaseLoader(url, bs_kwargs = {"parse_only":None})
  documents = loader.load()

  print(f"Loaded {len(documents)} documents")
  print(f"Source: {documents[0].metadata.get('source','N/A')}")
  print(f"Content length {len(documents[0].page_content)}")
  print(f"Content: {documents[0].page_content}")


if __name__ == "__main__":
  #load_text_file()
  #pdf_loader("C:/Users/JANARTHAN/lang-course/doc/langchain_demo.pdf")
  web_loader("https://en.wikipedia.org/wiki/Web_scraping")