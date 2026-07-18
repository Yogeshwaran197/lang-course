from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

document = """

1. What is Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a retriever with a generator. Instead of relying only on the parametric knowledge baked into an LLM's weights, RAG fetches relevant documents at query time and feeds them into the prompt as context. This reduces hallucination and lets you update knowledge without retraining the model.

A typical RAG pipeline has three stages: ingestion, retrieval, and generation. Ingestion covers loading documents, cleaning them, chunking them, embedding them, and storing them in a vector database. Retrieval covers converting a user query into an embedding and finding the nearest chunks. Generation covers stuffing those chunks into a prompt and asking an LLM to answer using them.

2. Why Chunking Matters

Chunking is the process of splitting a large document into smaller pieces before embedding. Chunk size directly affects retrieval quality. Chunks that are too large dilute the embedding with unrelated content, hurting precision. Chunks that are too small lose context, hurting recall, because a single chunk might not contain enough information to answer a question on its own.

There are two dominant chunking philosophies. Recursive character splitting divides text by a fixed set of separators (paragraph breaks, then sentences, then words) until each chunk is under a target token count. It is fast, deterministic, and ignores meaning entirely. Semantic chunking instead embeds sentences or small groups of sentences, measures the similarity between adjacent groups, and inserts a chunk boundary wherever similarity drops below a threshold. It is slower and non-deterministic across embedding models, but tends to keep topically coherent ideas together.

3. Coffee Brewing Methods

Pour-over coffee uses a gooseneck kettle to control water flow precisely over grounds in a paper filter. The Hario V60 is the most common dripper shape. Water temperature between 195°F and 205°F is recommended for most roasts.

French press brewing steeps coarse grounds directly in hot water for four minutes before pressing a metal mesh plunger through the slurry. Because there is no paper filter, more oils and fine sediment pass into the cup, giving a heavier body than pour-over.

Espresso forces hot water through finely ground, tightly packed coffee at roughly nine bars of pressure. A proper shot takes twenty-five to thirty seconds and produces a layer of crema on top. Espresso is the base for milk drinks like lattes, cappuccinos, and flat whites.

4. Vector Databases

A vector database stores embeddings alongside metadata and supports approximate nearest neighbor (ANN) search. Popular options include FAISS, Chroma, Qdrant, Pinecone, and Weaviate. FAISS is a library rather than a full database — it has no built-in persistence or metadata filtering unless you build that layer yourself. Chroma and Qdrant ship with metadata filtering and persistence out of the box, making them popular for local RAG prototyping.

Distance metrics matter. Cosine similarity measures the angle between two vectors and ignores magnitude, which is why most sentence embedding models are trained and evaluated using cosine distance. Euclidean (L2) distance measures straight-line distance and is sensitive to vector magnitude. Dot product is fastest to compute but only behaves like cosine similarity when vectors are normalized.

5. The Migratory Patterns of Monarch Butterflies

Monarch butterflies in North America undertake a multi-generational migration spanning up to 3,000 miles. Individuals born in late summer enter a state called reproductive diapause, which extends their lifespan from a few weeks to several months, allowing them to complete the journey south to overwintering sites in the oyamel fir forests of central Mexico.

No single butterfly completes the round trip. The generation that arrives in Mexico in November is not the same generation that starts the journey back north in spring; it takes three to four generations to repopulate the breeding range across the United States and southern Canada each year.

6. Hybrid Search

Hybrid search combines sparse retrieval (like BM25, which scores documents on exact keyword overlap using term frequency and inverse document frequency) with dense retrieval (embedding-based semantic similarity). BM25 is excellent at matching rare terms, product codes, acronyms, and exact phrases that embedding models often smooth over. Dense retrieval is better at matching paraphrases and conceptually related text that shares no vocabulary with the query.

A common fusion method is Reciprocal Rank Fusion (RRF), which combines ranked lists from multiple retrievers by scoring each document as the sum of 1/(k + rank) across all lists it appears in, without needing to normalize raw similarity scores onto the same scale.

7. A Short Note on Sourdough Starters

A sourdough starter is a live culture of wild yeast and lactic acid bacteria fermenting a mixture of flour and water. It needs daily feeding at room temperature or weekly feeding if refrigerated. A mature starter should double in size within four to six hours of feeding and smell pleasantly sour, not like nail polish remover, which indicates it has gone hungry for too long.

8. Evaluating RAG Systems

RAG evaluation typically covers retrieval metrics and generation metrics separately. Retrieval metrics include precision@k, recall@k, and mean reciprocal rank (MRR), all of which require a labeled set of query-to-relevant-chunk mappings. Generation metrics include faithfulness (does the answer only state things supported by the retrieved context), answer relevance (does the answer address the question asked), and context relevance (were the retrieved chunks actually useful).

Frameworks like RAGAS automate these scores by using an LLM as a judge, prompting it to check each claim in the generated answer against the retrieved context and flag unsupported statements. This is cheaper than human evaluation but inherits whatever biases and blind spots the judge model has."""


recursive_chunk =  RecursiveCharacterTextSplitter(
  chunk_size = 400,
  chunk_overlap = 50,
  separators = ["\n\n","\n", ". ", " "]
)
recursive_chunker = recursive_chunk.split_text(document)

Semantic_chunk = SemanticChunker(
  embeddings, 
  breakpoint_threshold_type="percentile",
  breakpoint_threshold_amount=0.70
)
semantic_chunker = Semantic_chunk.split_text(document)

recursive_vectorstore = Chroma.from_texts(
  recursive_chunker,
  embeddings,
  collection_name="Recursive"
)

semantic_vectorstore = Chroma.from_texts(
  semantic_chunker,
  embeddings,
  collection_name="Semantic"
)


test_queries = [
  "what is python?",
  "why we use langchain in rag?",
  "why we use langgraph in rag?",
  "what is langsmith",
]


def retriveval(query, vectorstroe, name):
  """function that reterives"""
  result = vectorstroe.similarity_search(query, k=1)
  print(f"{name}\nquery:{query}")
  print(f"Content preview: {result[0].page_content[:150]}....\n\n")
  return result[0].page_content
 


def comparison():
  print("==" * 60)
  for query in test_queries:
    print(retriveval(query, recursive_vectorstore, "Recursive_Chunker"))

  print("==" * 60)
  for query in test_queries:
    print(retriveval(query, semantic_vectorstore, "Semantic_Chunker"))


if __name__ =="__main__":
  comparison()



  