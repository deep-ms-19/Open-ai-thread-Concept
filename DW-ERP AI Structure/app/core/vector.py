from app.core.llm import client
from tempfile import NamedTemporaryFile

LATEST_VECTOR_STORE_ID = "vs_69327cadf43481918220138f6fa35226"

def create_vector_store(raw_text: str):
    """Create vector store from extracted PDF text."""
    with NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write(raw_text)
        txt_path = f.name

    vs = client.vector_stores.create(name="PDF-RAG-Store")

    with open(txt_path, "rb") as f:
        client.vector_stores.files.upload_and_poll(
            vector_store_id=vs.id,
            file=f
        )

    return vs.id


def query_vector_store(vector_id: str, query: str):
    """Retrieve context from vector store."""
    result = client.vector_stores.search(
        vector_store_id=vector_id,
        query=query
    )
    if result.data:
        return "\n".join([c.content[0].text for c in result.data])
    return ""
