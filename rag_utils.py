import os
import json
import tempfile
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


# ==============================
# 🔐 Load Environment Variables
# ==============================
load_dotenv()

VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")

# Ensure vector store directory exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


# ==============================
# 🤖 Initialize Embeddings
# ==============================
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ==============================
# 📂 Process Uploaded Files
# ==============================
def process_files(files, chunk_size=1000, chunk_overlap=100):
    docs = []

    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Select loader based on file type
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_ext == ".csv":
            loader = CSVLoader(tmp_path)
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
        else:
            print(f"❌ Unsupported file type: {file_ext}")
            continue

        docs.extend(loader.load())

    if not docs:
        raise ValueError("No valid documents found!")

    # ✂️ Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # 💾 Store in vector DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    vectordb.persist()
    print("✅ Documents processed and stored successfully!")


# ==============================
# ❓ Ask Question (RAG)
# ==============================
def ask_question(query, k=3):
    vectordb = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # 🚀 Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa({"query": query})

    answer = result["result"]
    sources = result.get("source_documents", [])

    # Save history
    log_result(query, answer, sources)

    return answer, [doc.metadata for doc in sources]


# ==============================
# 📝 Save Conversation History
# ==============================
def log_result(query, answer, sources):
    entry = {
        "query": query,
        "answer": answer,
        "sources": [doc.metadata for doc in sources]
    }

    # Load existing history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.append(entry)

    # Save updated history
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ==============================
# 📜 Load Conversation History
# ==============================
def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []