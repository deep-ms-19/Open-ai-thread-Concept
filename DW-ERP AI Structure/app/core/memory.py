from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS
from app.core.llm import embeddings

session_store = {}

# Initialize long-term memory
longterm_memory = FAISS.from_texts(
    ["Hello, I’m your assistant."],
    embedding=embeddings
)

def get_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]
