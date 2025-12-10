import streamlit as st
from dotenv import load_dotenv
import os

# ------------------------
# LangChain + OpenAI
# ------------------------
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# ========================
# ENV + OPENAI
# ========================
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

st.set_page_config(page_title="RAG + Memory Chat", layout="wide")

# =====================================================
#              GLOBAL VECTOR STORE  (no upload)
# =====================================================
LATEST_VECTOR_STORE_ID = "vs_69327cadf43481918220138f6fa35226"

# =====================================================
#                MEMORY + LLM SETUP
# =====================================================

main_llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4.1-nano",
    max_tokens=300,
    openai_api_key=OPENAI_KEY
)

suggestion_llm = ChatOpenAI(
    temperature=0.5,
    model="gpt-4o-mini",
    max_tokens=60,
    openai_api_key=OPENAI_KEY
)

# Long-term FAISS memory
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
longterm_memory = FAISS.from_texts(["Hello, I’m your assistant."], embedding=embeddings)

# Short-term session memory
if "session_store" not in st.session_state:
    st.session_state.session_store = {}

def get_history(session_id):
    store = st.session_state.session_store
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ========================= PROMPT ============================
prompt = ChatPromptTemplate.from_messages([
     ("system",
      "You are a DW-ERP assistant. Only answer using vector store context. "
      "NEVER hallucinate. Be concise and complete."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | main_llm

chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)


# =====================================================
#                   CHAT FUNCTION
# =====================================================
def process_user_message(session_id, user_text):
    # Add to long-term memory
    longterm_memory.add_texts([user_text])
    docs = longterm_memory.similarity_search(user_text, k=3)
    memory_context = "\n".join([d.page_content for d in docs])

    # Vector Store search
    pdf_context = ""
    if LATEST_VECTOR_STORE_ID:
        results = client.vector_stores.search(
            vector_store_id=LATEST_VECTOR_STORE_ID,
            query=user_text
        )
        if results.data:
            pdf_context = "\n".join([c.content[0].text for c in results.data])

    full_input = f"""
User message: {user_text}

Long-term memory:
{memory_context}

PDF/RAG context:
{pdf_context}
"""

    ai_reply = chat_with_memory.invoke(
        {"input": full_input},
        config={"configurable": {"session_id": session_id}}
    ).content

    # Generate suggestions
    suggestion_prompt = f"""
    Based on the user question:
    "{user_text}"

    Generate 3 helpful follow-up questions.
    ONLY return the list.
    """
    sug_res = suggestion_llm.invoke(suggestion_prompt)
    suggestions = [s.strip("-• ").strip() for s in sug_res.content.split("\n") if s.strip()][:3]

    return ai_reply, suggestions


# =====================================================
#                    STREAMLIT UI
# =====================================================

st.title("DW-ERP AI Assistant")

session_id = "default-session"

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation
for index, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

            # ---------- Suggestions below AI reply ----------
            if msg.get("suggestions"):
                st.markdown("### 🔍 Suggested Follow-up Questions")

                cols = st.columns(len(msg["suggestions"]))
                for i, s in enumerate(msg["suggestions"]):

                    if cols[i].button(s, key=f"suggestion_{index}_{i}"):

                        # Auto-send suggestion as user message
                        st.session_state.messages.append({"role": "user", "content": s})

                        ai_reply, next_suggestions = process_user_message(session_id, s)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ai_reply,
                            "suggestions": next_suggestions
                        })

                        st.rerun()


# ---------- User manual input ----------
user_text = st.chat_input("Ask something...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    ai_reply, suggestions = process_user_message(session_id, user_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply,
        "suggestions": suggestions
    })

    st.rerun()
