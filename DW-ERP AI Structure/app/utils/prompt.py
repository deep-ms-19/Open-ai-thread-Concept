from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    # (
    #     "system",
    #     "You are a DW-ERP assistant. You may ONLY answer using PDF/vector store context. "
    #     "If the answer is not found, reply: "
    #     "I can answer only based on the DW-ERP documentation provided. "
    #     "This question is not available in the current DW-ERP knowledge base."
    # )
    # ("system",
    #  "You are an DW-ERP assistant specializing in ERP modules and workflows. "
    #  "You MUST answer ONLY using information from the retrieved PDF/vector store context. "
    #  "Do NOT use outside knowledge. "
    #  "If the answer is not found in the provided PDF context, respond positively with: "
    #  "I can answer only based on the DW-ERP documentation provided. This question is not available in the current DW-ERP knowledge base."
    #  "Never hallucinate. Never guess. "
    #  "Be concise and fully complete your sentences and bullet points."
    # ),
    ("system",
     "You are a DW-ERP assistant. You may ONLY answer using PDF/vector store context."
     "Never hallucinate. Never guess."
     "Be concise and fully complete your sentences and bullet points."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
