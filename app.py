import os
import sys
from datetime import datetime
from typing import Generator, List, Dict, Any
import streamlit as st
from dotenv import load_dotenv
import ollama

# ------------------------------------------------------------------------------
# Configuration & Environment
# ------------------------------------------------------------------------------
load_dotenv()

def get_setting(name: str, default: str) -> str:
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value.strip()
    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None
    return str(secret_value).strip() if secret_value else default


DEFAULT_HOST = get_setting("OLLAMA_HOST", "https://ollama.com").rstrip("/")
DEFAULT_MODEL = get_setting("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_API_KEY = get_setting("OLLAMA_API_KEY", "")
IS_CLOUD_HOST = DEFAULT_HOST.lower() in {"https://ollama.com", "https://api.ollama.com"}

st.set_page_config(
    page_title="Enterprise AI Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Professional Enterprise Styling (Zero Icons / Zero Emojis)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Suppress default avatars and icon glyphs completely */
    [data-testid="stChatMessageAvatar"] {
        display: none !important;
    }
    [data-testid="stAlert"] [data-testid="stAlertIcon"],
    [data-testid="stNotification"] [data-testid="stNotificationIcon"] {
        display: none !important;
    }
    
    /* Completely remove all menu bar items: toolbar, hamburger menu, deploy button, status, decoration, footer */
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    .stDeployButton,
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Professional clean layout & typography */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
    }
    
    .main-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin: 0;
    }
    .main-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    
    /* Chat message boxes */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    
    .role-badge {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.35rem;
    }
    .role-user {
        color: #0369a1;
    }
    .role-assistant {
        color: #0f172a;
    }
    .message-timestamp {
        font-size: 0.7rem;
        font-weight: 400;
        color: #94a3b8;
    }
    
    /* Status banner */
    .status-panel {
        font-size: 0.82rem;
        padding: 0.55rem 0.75rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        border-width: 1px;
        border-style: solid;
    }
    .status-online {
        background-color: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }
    .status-offline {
        background-color: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Client & Model Service
# ------------------------------------------------------------------------------
@st.cache_resource
def get_ollama_client(host: str, api_key: str) -> ollama.Client:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return ollama.Client(host=host, headers=headers)

client = get_ollama_client(DEFAULT_HOST, OLLAMA_API_KEY)

def check_connection(c: ollama.Client) -> bool:
    try:
        c.list()
        return True
    except Exception:
        return False

def get_available_models(c: ollama.Client) -> List[str]:
    try:
        response = c.list()
        return [m.model for m in response.models]
    except Exception:
        return []

is_online = bool(OLLAMA_API_KEY) if IS_CLOUD_HOST else check_connection(client)
if is_online:
    is_online = check_connection(client)
available_models = get_available_models(client) if is_online else []

# Fallback model list if none fetched
if not available_models:
    available_models = [DEFAULT_MODEL]

# ------------------------------------------------------------------------------
# Session State Initialization & Background Configuration
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

selected_model = DEFAULT_MODEL

system_prompt = (
    "You are a professional executive advisor. Provide concise, structured, and strategic answers. "
    "Prioritize business value, actionable insights, and rigorous clarity. Avoid informal language."
)
temperature = 0.7
top_p = 0.9

# ------------------------------------------------------------------------------
# Main Content Area
# ------------------------------------------------------------------------------
header_col, action_col = st.columns([3, 1], gap="medium")

with header_col:
    st.markdown(
        """
        <div class="main-header">
            <h1 class="main-title">Enterprise Language Model Workspace</h1>
            <div class="main-subtitle">Ollama Cloud Inference Session</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with action_col:
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    with col_export:
        transcript_text = "# Conversation Transcript\n\n"
        transcript_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        transcript_text += f"Model: {selected_model}\n\n---\n\n"
        for msg in st.session_state["messages"]:
            transcript_text += f"### {msg['role'].upper()} [{msg.get('time', '')}]\n\n{msg['content']}\n\n"
            
        st.download_button(
            label="Export Chat",
            data=transcript_text,
            file_name=f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=(len(st.session_state["messages"]) == 0),
        )

# Render Chat History
for message in st.session_state["messages"]:
    role = message["role"]
    content = message["content"]
    timestamp = message.get("time", "")
    
    with st.chat_message(role):
        role_label_class = "role-user" if role == "user" else "role-assistant"
        role_display = "USER" if role == "user" else f"ASSISTANT ({selected_model})"
        
        st.markdown(
            f"""
            <div class="role-badge {role_label_class}">
                <span>{role_display}</span>
                <span class="message-timestamp">{timestamp}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(content)

# ------------------------------------------------------------------------------
# Inference Execution Handler
# ------------------------------------------------------------------------------
def generate_stream(
    c: ollama.Client,
    model: str,
    dialogue: List[Dict[str, str]],
    sys_instruction: str,
    temp: float,
    p_val: float,
) -> Generator[str, None, None]:
    full_payload = [{"role": "system", "content": sys_instruction}] + dialogue
    response_stream = c.chat(
        model=model,
        messages=full_payload,
        stream=True,
        options={
            "temperature": temp,
            "top_p": p_val,
        },
    )
    for chunk in response_stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield delta

# Chat Input & Response Loop
if user_input := st.chat_input("Enter your prompt or instruction..."):
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Append and render user message
    st.session_state["messages"].append(
        {"role": "user", "content": user_input, "time": current_time}
    )
    
    with st.chat_message("user"):
        st.markdown(
            f"""
            <div class="role-badge role-user">
                <span>USER</span>
                <span class="message-timestamp">{current_time}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(user_input)
    
    # Generate and stream assistant message
    if not is_online:
        if IS_CLOUD_HOST and not OLLAMA_API_KEY:
            st.error("Error: Set OLLAMA_API_KEY in the .env file to connect to Ollama Cloud.")
        else:
            st.error("Error: Ollama service is not reachable at the configured host address.")
    else:
        with st.chat_message("assistant"):
            reply_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(
                f"""
                <div class="role-badge role-assistant">
                    <span>ASSISTANT ({selected_model})</span>
                    <span class="message-timestamp">{reply_time}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Prepare conversation history for the API call
            conversation_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"]
            ]
            
            try:
                stream_generator = generate_stream(
                    client,
                    selected_model,
                    conversation_payload,
                    system_prompt,
                    temperature,
                    top_p,
                )
                full_response = st.write_stream(stream_generator)
                
                # Append assistant reply to state
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "time": reply_time,
                    }
                )
            except Exception as e:
                error_msg = f"Inference execution failed: {str(e)}"
                st.markdown(f'<div class="status-panel status-offline">{error_msg}</div>', unsafe_allow_html=True)
