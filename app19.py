# ===========================================================
# Unified RAG + Long-Term Memory + Chat Memory (SDK 2.8.1)
# Patched Version (100% thread-id safe)
# ===========================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import tempfile
import fitz

# ------------------------
# OpenAI + LangChain (FAISS only)
# ------------------------
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="Unified RAG + Memory Chat API (SDK 2.8.1)")

# -----------------------------------------------------------
# CORS
# -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================
# GLOBAL STATE
# ===========================================================
LATEST_VECTOR_STORE_ID = "vs_69327cadf43481918220138f6fa35226"
THREADS = {}  # session_id → thread_id

# ===========================================================
# OCR USING GPT-4o-mini
# ===========================================================
def ocr_image_with_gpt4(image_bytes):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "Extract all text. Convert tables into Markdown."},
            {"role": "user", "content": [{"type": "input_image", "image": image_bytes}]}
        ],
    )
    return response.output_text


# ===========================================================
# PDF EXTRACTION
# ===========================================================
def extract_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            final_text.append(f"### Page {i+1} - Text ###\n{text}")

        # OCR images
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_img = doc.extract_image(xref)
            image_bytes = base_img["image"]

            try:
                ocr_text = ocr_image_with_gpt4(image_bytes)
                final_text.append(f"### Page {i+1} - OCR {idx+1} ###\n{ocr_text}")
            except Exception as e:
                final_text.append(f"[OCR FAILED: {str(e)}]")

    return "\n\n".join(final_text)


# ===========================================================
# UPLOAD TO VECTOR STORE
# ===========================================================
def upload_to_vector_store(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write(text)
        temp_path = f.name

    vs = client.vector_stores.create(name="PDF-RAG-Store")

    with open(temp_path, "rb") as f:
        client.vector_stores.files.upload_and_poll(
            vector_store_id=vs.id,
            file=f
        )

    return vs.id


# ===========================================================
# PDF UPLOAD ENDPOINT
# ===========================================================
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global LATEST_VECTOR_STORE_ID

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_pdf_content(temp_path)

    vector_store_id = upload_to_vector_store(extracted_text)
    LATEST_VECTOR_STORE_ID = vector_store_id

    return {
        "message": "PDF processed successfully.",
        "vector_store_id": vector_store_id
    }


# ===========================================================
# FAISS LONG-TERM MEMORY
# ===========================================================
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
longterm_memory = FAISS.from_texts(
    ["Hello, I'm your assistant."],
    embedding=embeddings
)

# ===========================================================
# CHAT MODELS
# ===========================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    user_query: str
    response: str
    suggestions: list[str]
    thread_id: str  # ALWAYS returned now


# ===========================================================
# PATCHED /chat ENDPOINT (error-safe thread_id)
# ===========================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global LATEST_VECTOR_STORE_ID

    session_id = req.session_id
    user_input = req.message
    thread_id = None  # safe default

    try:
        # 1️⃣ Ensure thread exists
        if session_id not in THREADS:
            new_thread = client.beta.threads.create()
            print("\nTHREAD CREATED:", new_thread)  # Debug
            THREADS[session_id] = new_thread.id

        thread_id = THREADS[session_id]

        # 2️⃣ Long-term memory
        longterm_memory.add_texts([user_input])
        docs = longterm_memory.similarity_search(user_input, k=3)
        memory_context = "\n".join([d.page_content for d in docs])

        # 3️⃣ PDF RAG search
        pdf_context = ""
        if LATEST_VECTOR_STORE_ID:
            result = client.vector_stores.search(
                vector_store_id=LATEST_VECTOR_STORE_ID,
                query=user_input
            )
            if result.data:
                pdf_context = "\n".join([c.content[0].text for c in result.data])

        # 4️⃣ System instructions
        system_context = f"""
Use the following context ONLY if relevant.

Long-term memory:
{memory_context}

PDF search results:
{pdf_context}
"""

        # 5️⃣ Add user message to thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )

        # 6️⃣ Run model without assistant
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            model="gpt-4o-mini",
            instructions=system_context
        )

        # 7️⃣ Fetch messages
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        print("\nMESSAGES LIST:", messages)  # Debug

        # Safe extraction
        bot_reply = "I'm sorry, I couldn't generate a response."

        for msg in reversed(messages.data):
            if msg.role == "assistant":
                if msg.content and hasattr(msg.content[0], "text"):
                    bot_reply = msg.content[0].text.value
                else:
                    bot_reply = str(msg.content)
                break

        # 8️⃣ Follow-up suggestions
        sug = client.responses.create(
            model="gpt-4o-mini",
            input=f"""
User said:
\"{user_input}\"

Generate 3 helpful follow-up questions.
Return ONLY the questions.
"""
        )

        suggestions = [
            s.strip("-• ").strip()
            for s in sug.output_text.split("\n")
            if s.strip()
        ][:3]

    except Exception as e:
        # 🔥 NEVER let the endpoint fail
        bot_reply = f"Internal error: {str(e)}"
        suggestions = []

    # ALWAYS return thread_id no matter what
    print("\nRETURNING THREAD ID:", thread_id)

    return ChatResponse(
        session_id=session_id,
        user_query=user_input,
        response=bot_reply,
        suggestions=suggestions,
        thread_id=str(thread_id)
    )


# ===========================================================
# ROOT
# ===========================================================
@app.get("/")
async def root():
    return {"message": "Unified RAG + Memory API (SDK 2.8.1) is running!"}
