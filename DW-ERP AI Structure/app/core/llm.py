from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.core.settings import OPENAI_KEY

client = OpenAI(api_key=OPENAI_KEY)

main_llm = ChatOpenAI(
    temperature=0.3,
    # model="gpt-4o-mini",
    model="gpt-4.1-nano",
    openai_api_key=OPENAI_KEY
)

suggestion_llm = ChatOpenAI(
    temperature=0.5,
    model="gpt-4o-mini",
    max_tokens=60,
    openai_api_key=OPENAI_KEY
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
