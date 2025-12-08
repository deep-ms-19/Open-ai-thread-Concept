# ===========================================================
# Unified RAG + Long-Term Memory + Chat Memory (SDK 2.8.1) RAM stored 
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

# session_id → list of messages: [{"role":"user"...}, {"role":"assistant"...}]
CHAT_HISTORY = {}

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
# CHAT ENDPOINT
# ===========================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str



class ChatResponse(BaseModel):
    session_id: str
    user_query: str
    response: str
    suggestions: list[str]


def get_history(session_id: str):
    """Return history list; create if missing."""
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    return CHAT_HISTORY[session_id]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global LATEST_VECTOR_STORE_ID

    user_input = req.message
    session_id = req.session_id

    # 1️⃣ Retrieve session chat memory
    history = get_history(session_id)

    # 2️⃣ Long-term FAISS memory search
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

    # 4️⃣ Build system context
    system_context = f"""
Use the following context ONLY if relevant.

Long-term memory:
{memory_context}

PDF search results:
{pdf_context}
"""

    # 5️⃣ Add current user message to history
    history.append({"role": "user", "content": user_input})

    # 6️⃣ Run model with full conversation history
    messages = [{"role": "system", "content": system_context}] + history

    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages
    )

    bot_reply = response.output_text

    # Save bot reply to history
    history.append({"role": "assistant", "content": bot_reply})

    # 7️⃣ Generate suggestions
    sug_prompt = f"""
User said:
"{user_input}"

Generate 3 useful follow-up questions.
Return ONLY the questions.
"""

    sug = client.responses.create(
        model="gpt-4o-mini",
        input=sug_prompt
    )

    suggestions = [
        s.strip("-• ").strip()
        for s in sug.output_text.split("\n")
        if s.strip()
    ][:3]

    
    return ChatResponse(
    session_id=session_id,
    user_query=user_input,
    response=bot_reply,
    suggestions=suggestions
)


# ===========================================================
# ROOT
# ===========================================================
@app.get("/")
async def root():
    return {"message": "Unified RAG + Memory API (SDK 2.8.1) is running!"}
