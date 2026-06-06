
import warnings
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from google import genai
from duckduckgo_search import DDGS

# ==========================================
# CONFIG
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

app = FastAPI()

# ==========================================
# MEMORY
# ==========================================

conversation_history = {}

# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):
    user_id: str
    message: str

# ==========================================
# WEB SEARCH
# ==========================================

def web_search(query):

    try:

        with DDGS() as ddgs:

            results = list(
                ddgs.text(
                    query,
                    max_results=5
                )
            )

        text = ""

        for r in results:

            title = r.get("title", "")
            body = r.get("body", "")

            text += f"""
Title: {title}
Snippet: {body}

"""

        return text

    except Exception as e:

        return f"Search unavailable: {str(e)}"

# ==========================================
# CHAT FUNCTION
# ==========================================

def generate_answer(user_id, question):

    if user_id not in conversation_history:

        conversation_history[user_id] = []

    history = conversation_history[user_id]

    search_results = web_search(question)

    history.append(
        f"User: {question}"
    )

    history_text = "\n".join(history[-20:])

    prompt = f"""
You are an intelligent AI assistant.

Use conversation history to answer
follow-up questions naturally.

If web search information is useful,
use it.

Conversation History:

{history_text}

Web Search Results:

{search_results}

Answer the user's question.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    history.append(
        f"Assistant: {answer}"
    )

    return answer

# ==========================================
# API
# ==========================================

@app.post("/chat")
def chat(req: ChatRequest):

    answer = generate_answer(
        req.user_id,
        req.message
    )

    return {
        "answer": answer
    }

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
def home():

    return {
        "status": "running"
    }
# if __name__ == "__main__":

#     user = "demo"

#     print(
#         generate_answer(
#             user,
#             "Who is Elon Musk?"
#         )
#     )

#     print(
#         generate_answer(
#             user,
#             "How old is he?"
#         )
#     )

#     print(
#         generate_answer(
#             user,
#             "Latest Tesla news"
#         )
#     )

