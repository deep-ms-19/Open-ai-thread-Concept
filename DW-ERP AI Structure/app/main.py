from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, upload

app = FastAPI(title="Unified RAG + Memory Chat API")

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Routers
# ---------------------------
app.include_router(upload.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Unified RAG + Memory API is running!"}
