# live_data.py
import os
from dotenv import load_dotenv
import requests
import feedparser
import yfinance as yf
from google import genai
from duckduckgo_search import DDGS

# Configuration via environment variables with your fallback
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
client = genai.Client(api_key=GEMINI_API_KEY)

def weather_tool(city):
    try:
        data = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10).json()
        current = data["current_condition"][0]
        return {
            "city": city,
            "temperature_c": current["temp_C"],
            "humidity": current["humidity"],
            "condition": current["weatherDesc"][0]["value"]
        }
    except Exception as e:
        return {"error": str(e)}

def stock_tool(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "company": info.get("longName"),
            "symbol": symbol,
            "price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE")
        }
    except Exception as e:
        return {"error": str(e)}

def news_tool(topic):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={topic}")
        return [item.title for item in feed.entries[:5]]
    except Exception as e:
        return {"error": str(e)}

def search_tool(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        text = ""
        for r in results:
            text += f"\nTitle: {r.get('title')}\nSnippet: {r.get('body')}\n"
        return text if text.strip() else "No web results found."
    except Exception as e:
        return f"Search anomaly: {str(e)}"

def decide_tool(question, history_context):
    prompt = f"""You are an AI routing engine.

Conversation History:
{history_context}

Current User Question:
{question}

Available tools:
1. weather(city)
2. stock(symbol)
3. news(topic)
4. search(query)

Choose ONE tool that best fits the request. 
Return ONLY one line formatted exactly as tool:argument (e.g., weather:Paris or search:latest AI news).

No explanation."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()

def ask_with_history(question, history_context):
    tool_choice = decide_tool(question, history_context)
    print(f"🛠️ [Live Sub-Tool Chosen] -> {tool_choice}")

    context = ""
    if tool_choice.startswith("weather:"):
        context = str(weather_tool(tool_choice.replace("weather:", "").strip()))
    elif tool_choice.startswith("stock:"):
        context = str(stock_tool(tool_choice.replace("stock:", "").strip()))
    elif tool_choice.startswith("news:"):
        context = str(news_tool(tool_choice.replace("news:", "").strip()))
    else:
        query = tool_choice.replace("search:", "").strip() if tool_choice.startswith("search:") else tool_choice
        context = search_tool(query)

    final_prompt = f"""You are an intelligent AI assistant capable of real-time search lookup.

Conversation History:
{history_context}

User Question: {question}

Retrieved Real-Time Information:
{context}

Instructions:
- Seamlessly answer using the retrieved information.
- Maintain natural conversation and context flow.
- Be concise but highly informative."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt
    )
    return response.text