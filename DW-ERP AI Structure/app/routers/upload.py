from fastapi import APIRouter, UploadFile, File
from app.utils.pdf import extract_pdf_text
from app.core.vector import create_vector_store, LATEST_VECTOR_STORE_ID

router = APIRouter(prefix="/upload", tags=["PDF"])

@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global LATEST_VECTOR_STORE_ID

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    extracted = extract_pdf_text(temp_path)
    vector_id = create_vector_store(extracted)

    LATEST_VECTOR_STORE_ID = vector_id

    return {
        "message": "PDF processed successfully.",
        "vector_store_id": vector_id
    }
