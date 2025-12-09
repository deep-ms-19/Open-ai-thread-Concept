from fastapi import APIRouter
from pydantic import BaseModel

from app.core.llm import main_llm, suggestion_llm
from app.core.memory import longterm_memory
from app.core.vector import LATEST_VECTOR_STORE_ID, query_vector_store

from app.utils.prompt import prompt
from app.core.memory import get_history
from langchain_core.runnables.history import RunnableWithMessageHistory

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    session_id: str
    user_query: str

class ChatResponse(BaseModel):
    session_id: str
    user_query: str
    response: str
    suggestions: list[str]

chain = prompt | main_llm

chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):

    user_input = req.user_query

    # Save to long-term memory
    longterm_memory.add_texts([user_input])

    # Retrieve memory
    docs = longterm_memory.similarity_search(user_input, k=3)
    memory_context = "\n".join([d.page_content for d in docs])

    # Retrieve PDF context
    pdf_context = query_vector_store(LATEST_VECTOR_STORE_ID, user_input) if LATEST_VECTOR_STORE_ID else ""

    # ---------- REMOVED: ERP KEYWORD VALIDATION ----------
    # No filtering, no blocking, no ERP keyword checks.

    final_input = f"""
User message: {user_input}

Long-term memory:
{memory_context}

PDF context:
{pdf_context}
"""

    # Generate answer
    response = chat_with_memory.invoke(
        {"input": final_input},
        config={"configurable": {"session_id": req.session_id}}
    ).content

    # Suggestions
    sug = suggestion_llm.invoke(f"""
Based on user question:
"{user_input}"

Generate 3 helpful follow-up questions.
Return ONLY the list.
""").content.split("\n")

    suggestions = [s.strip("-• ").strip() for s in sug if s.strip()][:3]

    return ChatResponse(
        session_id=req.session_id,
        user_query=user_input,
        response=response,
        suggestions=suggestions
    )
