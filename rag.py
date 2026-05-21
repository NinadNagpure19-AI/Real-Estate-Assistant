from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# UnstructuredURLLoader is the standard LangChain loader for URLs.
# It works well on Linux/Mac but requires native C libraries (libmagic, onnxruntime)
# that cause segfaults on Windows. As a Windows-safe alternative, we use
# requests + BeautifulSoup which produces identical LangChain Document objects
# and feeds into the same downstream RAG pipeline without any changes.
# from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from prompt import PROMPT

load_dotenv()

# Constants
CHUNK_SIZE = 1000

VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

HEADERS = {"User-Agent": "Mozilla/5.0"}


def load_urls(urls):
    """
    Fetch and parse URLs using requests + BeautifulSoup.
    Returns a list of LangChain Document objects — same format as UnstructuredURLLoader.
    """
    docs = []
    for url in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove noise tags
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            docs.append(Document(page_content=f"Failed to load {url}: {e}", metadata={"source": url}))
    return docs


def initialize_components():
    """
    Lazy initialize LLM and vector database.
    """

    global llm, vector_store

    # Initialize LLM only once
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500
        )

    # Initialize vector DB only once
    if vector_store is None:

        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    """
    Scrape URLs and store chunks inside vector DB.
    """

    yield "Initializing Components...✅"

    initialize_components()

    yield "Resetting vector database...✅"

    vector_store.reset_collection()

    yield "Loading webpage data...✅"

    # loader = UnstructuredURLLoader(urls=urls, headers=HEADERS)
    # data = loader.load()
    data = load_urls(urls)

    yield "Splitting text into chunks...✅"

    text_splitter = RecursiveCharacterTextSplitter(    # can be changed with token splitter, semantic chunker, markdown splitter, recursive json splitter
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )

    docs = text_splitter.split_documents(data)

    yield "Creating embeddings and storing vectors...✅"

    uuids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(docs, ids=uuids)

    yield "Documents processed successfully...✅"


def format_docs(docs):
    """
    Convert retrieved Document objects into one large context string.
    """
    # stuffs all chunks together.
    return "\n\n".join(doc.page_content for doc in docs)  # can also be replaced by reranking, map reduce, refine, compression, contextual filtering


def generate_answer(query):
    """
    Generate answer using retrieval + prompt + LLM pipeline.
    """

    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    # Create retriever object
    retriever = vector_store.as_retriever(
        search_type="similarity"             # "mmr"(Marginal Maximum Relevance); "similarity_score_threshold"
    )
                                            # similarity -> most relevant chunks
                                            # mmr -> relevance + diversity
                                            # similarity_score_threshold -> retrieve only above score threshold

    # LCEL Retrieval Pipeline (LangChain Expression Language)
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT | llm | StrOutputParser()
    )

    answer = chain.invoke(query)

    # Retrieve source docs separately
    source_docs = retriever.invoke(query)

    sources = list(set(
        doc.metadata.get("source", "Unknown")
        for doc in source_docs
    ))

    return answer, sources


if __name__ == "__main__":

    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer(
        "Tell me what was the 30 year fixed mortgage rate along with the date?"
    )

    print(f"\nAnswer:\n{answer}")

    print(f"\nSources:\n{sources}")