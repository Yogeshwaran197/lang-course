import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ParentDocumentRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

llm =  ChatGroq(
    model ="openai/gpt-oss-120b",
    api_key= os.environ["GROQ_API_KEY"]
)

embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3",
)

def chunk(document):

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    return [c.strip() for c in splitter.split_text(document)]


def contextual_retrieval(document: str):

    chunks = chunk(document)
    prompt = ChatPromptTemplate.from_template(
        """<document>
        {document}
        </document>
 
        Here is the chunk we want to situate within the whole document:
 
        <chunk>
        {chunk}
        </chunk>
 
        Please give a short, succinct context to situate this chunk within the
        overall document, for the purpose of improving search retrieval of this chunk.
        Answer only with the succinct context, and nothing else."""

    ) 

    chain =  prompt  | llm | StrOutputParser()

    contextual_docs = []

    for c in  chunks:
        context = chain.invoke({"document": document, "chunk": c})
        contextual_docs.append(Document(page_content=f"{context}\n\n{c}"))

    return contextual_docs

def ensemble_retriever(document, query):

    if isinstance(document, str):
        docs = [Document(page_content=document)]
    else:
        docs = document

    bm25_retriever = BM25Retriever.from_documents(
        docs,
        k = 4
    )

    vectorstore = Chroma.from_documents(
        docs,
        embedding = embeddings
    )

    vector_reteriever = vectorstore.as_retriever(
        search_kwargs = {
            "k": 4
        }
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, vector_reteriever],
        weights= [0.5, 0.5]
    )

    top_results = ensemble_retriever.invoke(query)[:4]

    return top_results

def format_doc(doc):
    return "\n\n".join(d.page_content for d in doc)



def run_pipeline(document, query):

    top_results = ensemble_retriever(document, query)
    context = format_doc(top_results)

    prompt = ChatPromptTemplate.from_template(
        """you're an helpful AI assistant answer the question ONLY
        using the {context} for given question.

        question:{question}

    """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"question":query, "context":context})

    return response



if __name__ == "__main__":

    full_document  = """
    Apple Inc. Q2 2024 Earnings Report
    
    Section 1: Overview
    Apple Inc. today announced financial results for its fiscal 2024 second quarter
    ended March 30, 2024. The Company posted quarterly revenue of $90.8 billion, up
    3 percent year over year, and quarterly earnings per diluted share of $1.53, up
    16 percent year over year. Services revenue reached an all-time high, driven by
    strong growth in the App Store, Apple Music, iCloud, and AppleCare. iPhone
    revenue was roughly flat compared to the same quarter last year, while demand
    in emerging markets such as India and Southeast Asia continued to grow at a
    double-digit pace.
    
    Section 2: Revenue by Product Category
    iPhone revenue for the quarter came in at $45.9 billion, largely unchanged from
    the prior year period. Mac revenue grew to $7.5 billion, an increase driven by
    the transition to Apple Silicon across the lineup and strong back-to-school
    demand in international markets. iPad revenue declined to $5.6 billion as the
    category faced a difficult comparison against last year's iPad Pro launch.
    Wearables, Home and Accessories revenue was $7.9 billion, roughly flat, with
    strength in Apple Watch offset by softer AirPods sales. Services revenue, which
    includes the App Store, Apple Music, Apple TV+, iCloud, and AppleCare, grew to
    $23.9 billion, up 11 percent year over year and representing an increasing
    share of total company revenue.
    
    Section 3: Revenue by Geographic Segment
    The Americas segment generated $37.3 billion in revenue, up slightly from the
    prior year. Europe delivered $24.1 billion, supported by strong iPhone upgrade
    activity following recent carrier promotions. Greater China revenue declined to
    $16.4 billion amid intensifying competition from domestic smartphone makers and
    a broader slowdown in consumer spending. Japan posted revenue of $6.8 billion,
    roughly flat year over year. Rest of Asia Pacific, which includes India and
    Southeast Asia, was the fastest-growing region, posting revenue of $6.2 billion,
    up 8 percent year over year, with India setting a new all-time revenue record
    for the quarter.
    
    Section 4: Profitability and Margins
    Gross margin for the quarter was 46.6 percent, compared to 44.3 percent in the
    same quarter last year, reflecting a favorable mix shift toward higher-margin
    Services revenue and cost efficiencies in product manufacturing. Operating
    expenses were $14.4 billion, up modestly from the prior year, driven primarily
    by continued investment in research and development related to generative AI
    initiatives and custom silicon development. Operating income was $27.9 billion,
    representing an operating margin of 30.7 percent.
    
    Section 5: Capital Return Program
    During the quarter, the Board of Directors declared a cash dividend of $0.24
    per share of common stock, payable in the following quarter. The Company also
    returned over $23 billion to shareholders through dividends and share
    repurchases during the quarter, continuing its ongoing capital return program.
    The Board authorized an additional $110 billion for share repurchases, one of
    the largest authorizations in company history.
    
    Section 6: Outlook
    Management indicated that for the next quarter, the Company expects total
    revenue to grow at a low-to-mid single digit rate year over year, with Services
    revenue growth expected to remain in the low double digits. Management noted
    continued macroeconomic uncertainty in Greater China but expressed confidence
    in the long-term growth trajectory of emerging markets. The Company also
    reiterated its commitment to significant investment in artificial intelligence
    capabilities across its product lineup, with more details expected to be shared
    at the upcoming developer conference.
    """

    question =  "what is long term trajectory?"
    chunks = contextual_retrieval(full_document)
    docs = ensemble_retriever(document=chunks, query=question)
    results = run_pipeline(docs, query = question)

    print("="*60)
    print("Contextual Retrieval")
    print("="*60)

    print(f"\nQuery : {question}")
    print(results)
 
  

    

