""" Text Splitter module for splitting text into smaller chunks"""

from langchain_text_splitters import (
  RecursiveCharacterTextSplitter,
  CharacterTextSplitter,
  MarkdownHeaderTextSplitter,
  PythonCodeTextSplitter,
  Language,
)


SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

def Recursive_splitter():
  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators=["\n\n", "\n", " ", ""],
  )

  chunks = splitter.split_text(SAMPLE_TEXT)
  print("=== Recursive Character Splitter ===")
  print(f"Recursive Splitter Created {len(chunks)} chunks")
  print(f"Original Document Size: {len(SAMPLE_TEXT)}")
  print(f"Chunk sizes {[len(c) for c in chunks]}")
  print(f"Content preview:\n {chunks[0][:200]}")


def code_splitter():
  splitter = RecursiveCharacterTextSplitter.from_language(
      language=Language.PYTHON, 
      chunk_size = 500,
      chunk_overlap = 50
  )
  code_splitter = splitter.split_text(SAMPLE_CODE)
  print(f"Code Splitter created {len(code_splitter)} chunks")
  for i , code_splitter in enumerate(code_splitter):
    print(f"\n Chunk {1} ({len(code_splitter)} chars)")


def document_splitter():
  from langchain_community.document_loaders import PyPDFLoader
  from langchain_core.documents import Document

  loader = PyPDFLoader("./doc/langchain_demo.pdf")
  docs = loader.load()

  print(f"loaded {len(docs)} documents")

  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 75,
    separators = ["\n\n","\n"," ",""]
  )
  chunks = splitter.split_documents(docs)

  print(f"Document Splitter created {len(chunks)} chunks")
  for i , doc in enumerate(chunks):
    print(f"chunk {i + 1} :\n content preview: {doc.page_content[:500]}")

if __name__ == "__main__":
  #Recursive_splitter()
  #code_splitter()
  document_splitter()

