import fitz

def extract_pdf_text(pdf_path: str) -> str:
    """Extract plain page-wise text from a PDF."""
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            text = "[NO TEXT ON PAGE]"
        pages.append(f"### Page {i+1} ###\n{text}")

    return "\n\n".join(pages)
