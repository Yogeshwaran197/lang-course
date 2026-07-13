from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")

def basic_embedding():

  text = "what is python?"
  vector = embeddings.embed_query(text)

  print(f"Embedding for single text: {vector[:10]}")
  print(f"Dimension of vector: {len(vector)} dimensions")


def batch_embedding():
   embeds = embeddings.embed_documents(
     ["This is the first document.", "This is the second document."]
 )
   for i , embds in enumerate(embeds):
     print(f"Text - {i+1}:\n First 10 vectors {embds[:10]}")
     print(f"Dimension of vector: {len(embds)} dims")



def similarity_search():

  docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
  ]

  query = "what is  python exist?"

  doc_vec = embeddings.embed_documents(docs)
  query_vec = embeddings.embed_query(query)

  def cosine_similarity(vec1, vec2):
    return np.dot(vec1,vec2)/ np.linalg.norm(vec1) * np.linalg.norm(vec2)

  similarites = [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vec]

  ranked_doc = sorted(zip(docs, similarites), key =lambda x: x[1], reverse= True)
  print(f"Query : {query}\n")
  print("Ranked by Similarity Score")
  for doc , score in ranked_doc:
    print(f" score: {score:.4f}| {doc}")


if __name__ == "__main__":
  #basic_embedding()
  #batch_embedding()
  similarity_search()