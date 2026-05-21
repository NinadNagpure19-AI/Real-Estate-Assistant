from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant for Real Estate research.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
""")