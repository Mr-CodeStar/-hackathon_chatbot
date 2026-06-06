# api.py
from fastapi import FastAPI, Form, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import uuid
import os
import shutil
from database import init_db, get_db_connection

# Import internal submodules
import rag
import live_data

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    print("🛑 Shutting down application backend server.")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    # Clean up local persistent session vectors if needed on reset
    if os.path.exists(rag.VECTOR_STORE_DIR):
        shutil.rmtree(rag.VECTOR_STORE_DIR)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/history")
async def get_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, session_title FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    history_list = [{"session_id": row["session_id"], "session_title": row["session_title"]} for row in rows]
    conn.close()
    return {"history_list": history_list}

@app.get("/api/get_session_messages")
async def get_session_messages(session_id: str = Query(...)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sender, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    rows = cursor.fetchall()
    messages = [{"sender": row["sender"], "content": row["content"]} for row in rows]
    conn.close()
    return {"messages": messages}

@app.post("/api/send_prompt")
async def send_prompt(
    prompt: str = Form(...), 
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    print("\n============================ NEW ROUTER REQUEST ============================")
    print(f"📥 PROMPT: '{prompt}'")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    is_new_chat = False
    
    if not session_id or session_id in ["null", "undefined"]:
        session_id = str(uuid.uuid4())
        is_new_chat = True
        session_title = prompt if len(prompt) < 30 else prompt[:27] + "..."
        cursor.execute("INSERT INTO sessions (session_id, session_title) VALUES (?, ?)", (session_id, session_title))

    # Log User Message to Database
    cursor.execute("INSERT INTO messages (session_id, sender, content) VALUES (?, ?, ?)", (session_id, 'user', prompt))
    conn.commit() # Commit early so context injection includes the current message if fetched
    
    # Fetch conversation history from database for AI context injection
    cursor.execute("SELECT sender, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 10", (session_id,))
    past_rows = cursor.fetchall()
    formatted_history = []
    for r in reversed(past_rows):
        role = "User" if r["sender"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {r['content']}")
    history_context = "\n".join(formatted_history)

    agent_reply = ""

    # Check if a persistent document workspace index already exists for this chat session
    existing_vector_db = rag.load_vector_store_for_session(session_id)

    # ──────────────────────────────────────────────────────────────
    # MODE 1: DOCUMENT PROCESSING & PERSISTENT RAG WORKSPACE
    # ──────────────────────────────────────────────────────────────
    if (file and file.filename and file.filename.strip() != "") or (existing_vector_db is not None):
        
        # Scenario A: New file uploaded to this session
        if file and file.filename and file.filename.strip() != "":
            if not file.filename.lower().endswith('.pdf'):
                agent_reply = "⚠️ Please upload your document as a PDF file. Other formats are currently restricted."
            else:
                temp_file_path = os.path.join(TEMP_DIR, file.filename)
                try:
                    print(f"📁 Processing Uploaded PDF File: {file.filename}")
                    with open(temp_file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    # Process file and create index mapped to session_id
                    vector_db, documents = rag.process_and_save_document([temp_file_path], session_id)
                    
                    if "summarize" in prompt.lower() or len(prompt.strip()) < 5:
                        print("📝 Executing Document Summarization...")
                        agent_reply = rag.summarize_document(documents)
                    else:
                        print("🔍 Executing Document RAG Q&A...")
                        agent_reply = rag.answer_question(prompt, vector_db)
                except Exception as e:
                    print(f"💥 Document Processing Error: {e}")
                    agent_reply = f"Error evaluating document contents: {str(e)}"
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
        
        # Scenario B: Session follow-up query against an already uploaded document
        else:
            print("🔍 Contextual Continuous Session RAG Q&A...")
            try:
                agent_reply = rag.answer_question(prompt, existing_vector_db)
            except Exception as e:
                print(f"💥 Continuous RAG Error: {e}")
                agent_reply = f"Error evaluating session document workspace context: {str(e)}"

    # ──────────────────────────────────────────────────────────────
    # NO FILE & NO PREVIOUS SESSION WORKSPACE: Dynamic Search/General Router
    # ──────────────────────────────────────────────────────────────
    else:
        print("🤖 Invoking Intelligent Routing Engine Choice...")
        routing_prompt = f"""Determine if answering the following user request requires real-time information, current events, web search, weather data, news, or stock prices.

User Question: {prompt}

Respond with exactly ONE word:
"live" - If it requires web search, weather, news, stocks, or real-time info.
"general" - If it is a generic chat, calculation, code request, or historical knowledge.

Your response:"""
        
        try:
            route_res = live_data.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=routing_prompt
            )
            route_decision = route_res.text.strip().lower()
        except Exception:
            route_decision = "general"
        
        print(f"🔀 Routed Target Choice: [{route_decision.upper()}]")

        # MODE 2: LIVE ENGINE
        if "live" in route_decision:
            print("🌐 Executing Live Data Tool & Web Search Pipeline...")
            try:
                agent_reply = live_data.ask_with_history(prompt, history_context)
            except Exception as e:
                print(f"💥 Live Engine Error: {e}")
                agent_reply = f"Failed to retrieve real-time search context: {str(e)}"

        # MODE 3: GENERAL CHAT PIPELINE
        else:
            print("💬 Executing Standard Conversational LLM Pipeline...")
            try:
                response = live_data.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"System: You are an intelligent AI assistant. Use the conversation history below to seamlessly handle follow-ups.\n\nConversation History:\n{history_context}\n\nUser Question: {prompt}\n\nAnswer:"
                )
                agent_reply = response.text
            except Exception as e:
                print(f"💥 General Chat Connection Error: {e}")
                agent_reply = f"Failed to reach chat engine: {str(e)}"

    print("=====================================================================\n")

    # Save Assistant Response to Database
    cursor.execute("INSERT INTO messages (session_id, sender, content) VALUES (?, ?, ?)", (session_id, 'bot', agent_reply))
    conn.commit()
    
    # Get updated session listings for seamless UI updates
    cursor.execute("SELECT session_id, session_title FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    history_list = [{"session_id": r["session_id"], "session_title": r["session_title"]} for r in rows]
    conn.close()
    
    return {
        "session_id": session_id,
        "agent_response": agent_reply,
        "history_list": history_list,
        "is_new_chat": is_new_chat
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)