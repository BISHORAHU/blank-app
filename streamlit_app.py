import streamlit as st
import os
import re
import subprocess
import tempfile
import sqlite3
import asyncio
import logging
import yaml
from functools import lru_cache
from typing import Dict, Any

import streamlit.components.v1 as components

from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain_community.llms import Clarifai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the database
def init_database():
    conn = sqlite3.connect('chat_app.db')
    cursor = conn.cursor()
    # Create table for API keys
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL
        )
    ''')
    # Create table for chat history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_database()

# Set page configuration
st.set_page_config(layout="wide", page_title="Clarifai Chat Interface", page_icon="💬")
ClarifaiStreamlitCSS.insert_default_css(st)

# Load custom CSS
def load_css():
    css_file = './styles.css'
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"{css_file} not found. Skipping custom CSS.")

load_css()

# Load configuration from YAML file
@st.cache_resource
def load_config() -> Dict[str, Any]:
    config_file = 'config.yaml'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully.")
            return config
    else:
        logger.warning(f"{config_file} not found. Using default settings.")
        return {}

config = load_config()

# Initialize session state variables
def initialize_session_state():
    default_values = {
        'api_key': '',
        'chat_history': [],
        'message_count': 0,
        'conversation_id': 1,
        'code_execution_enabled': True,
        'memory': ConversationBufferMemory(ai_prefix="AI Assistant", memory_key="chat_history"),
        'conversation': None,
        'chosen_llm': None,
        'theme': 'default',
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Apply selected theme
def apply_theme(theme_name: str):
    themes = {
        'default': {
            'primaryColor': '#f63366',
            'backgroundColor': '#ffffff',
            'secondaryBackgroundColor': '#f0f2f6',
            'textColor': '#262730',
            'font': 'sans serif',
        },
        'dark': {
            'primaryColor': '#1f77b4',
            'backgroundColor': '#262730',
            'secondaryBackgroundColor': '#31333f',
            'textColor': '#fafafa',
            'font': 'sans serif',
        },
    }
    theme = themes.get(theme_name, themes['default'])
    st.write(f'<style>{get_css_for_theme(theme)}</style>', unsafe_allow_html=True)

def get_css_for_theme(theme: Dict[str, str]) -> str:
    return f"""
    .reportview-container {{
        background-color: {theme['backgroundColor']};
        color: {theme['textColor']};
        font-family: {theme['font']};
    }}
    .sidebar .sidebar-content {{
        background-color: {theme['secondaryBackgroundColor']};
    }}
    .stButton>button {{
        color: {theme['textColor']};
    }}
    """

apply_theme(st.session_state['theme'])

# Database functions for API keys
def save_api_key_to_db(api_key):
    conn = sqlite3.connect('chat_app.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO api_keys (key) VALUES (?)', (api_key,))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("API Key already exists in the database.")
    finally:
        conn.close()

def get_saved_api_keys():
    conn = sqlite3.connect('chat_app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT key FROM api_keys')
    keys = cursor.fetchall()
    conn.close()
    return [key[0] for key in keys]

# Database functions for chat history
def save_message_to_db(conversation_id, role, content):
    conn = sqlite3.connect('chat_app.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (conversation_id, role, content)
        VALUES (?, ?, ?)
    ''', (conversation_id, role, content))
    conn.commit()
    conn.close()

def load_chat_history_from_db(conversation_id):
    conn = sqlite3.connect('chat_app.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT role, content FROM chat_history
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    ''', (conversation_id,))
    messages = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in messages]

def get_conversation_ids():
    conn = sqlite3.connect('chat_app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT conversation_id FROM chat_history')
    ids = cursor.fetchall()
    conn.close()
    return [id[0] for id in ids]

# Sidebar for settings
st.sidebar.title("🔧 Settings")

# API Key management
def api_key_management():
    with st.sidebar.expander("API Key Management", expanded=True):
        api_key_input = st.text_input("Enter Clarifai API Key:", type="password")
        if st.button("Save API Key"):
            if api_key_input:
                save_api_key_to_db(api_key_input)
                st.session_state.api_key = api_key_input
                os.environ['CLARIFAI_PAT'] = st.session_state.api_key
                st.success("API Key saved and set successfully!")
            else:
                st.error("API Key is empty.")

        # List of saved API Keys
        saved_api_keys = get_saved_api_keys()
        if saved_api_keys:
            st.markdown("### Saved API Keys")
            selected_api_key = st.selectbox("Select an API Key", saved_api_keys)
            if st.button("Set Selected API Key"):
                st.session_state.api_key = selected_api_key
                os.environ['CLARIFAI_PAT'] = st.session_state.api_key
                st.success("API Key set successfully!")

api_key_management()

# Conversation Management
def conversation_management():
    with st.sidebar.expander("Conversation Management", expanded=False):
        conversation_ids = get_conversation_ids()
        if conversation_ids:
            selected_conversation = st.selectbox("Select a Conversation", conversation_ids)
            if st.button("Load Conversation"):
                st.session_state.conversation_id = selected_conversation
                st.session_state.chat_history = load_chat_history_from_db(selected_conversation)
                st.success(f"Conversation {selected_conversation} loaded.")
        else:
            st.info("No previous conversations found.")

conversation_management()

# Model selection
@lru_cache
def get_default_models():
    default_models = config.get('DEFAULT_MODELS', "o1-preview:openai;chat-completion, Llama-2:meta;llama")
    models_list = [x.strip() for x in default_models.split(",")]
    models_map = {}
    select_map = {}
    for m in models_list:
        id, rem = m.split(':')
        author, app = rem.split(';')
        models_map[id] = {'author': author, 'app': app}
        select_map[f"{id} : {author}"] = id
    return models_map, select_map

models_map, select_map = get_default_models()
default_llm = config.get('DEFAULT_LLM', "GPT-4")

llms_map = {'Select an LLM': None}
llms_map.update(select_map)

st.sidebar.markdown("### Model Selection")
chosen_llm = st.sidebar.selectbox("Select an LLM for chatting", options=llms_map.keys())
st.session_state.chosen_llm = llms_map.get(chosen_llm)

# Theme selection
st.sidebar.markdown("### Theme Selection")
theme_choice = st.sidebar.selectbox("Choose Theme", options=['default', 'dark'])
if theme_choice != st.session_state['theme']:
    st.session_state['theme'] = theme_choice
    apply_theme(theme_choice)

# Code execution toggle
st.session_state.code_execution_enabled = st.sidebar.checkbox(
    "Enable Code Execution",
    value=st.session_state.code_execution_enabled
)

# New chat button
if st.sidebar.button("🆕 New Chat"):
    st.session_state.chat_history = []
    st.session_state.message_count = 0
    st.session_state.conversation_id = max(get_conversation_ids() + [0]) + 1
    st.session_state.memory.clear()
    st.session_state.conversation = None
    st.experimental_rerun()

# Display message count and conversation ID
st.sidebar.markdown("---")
st.sidebar.write(f"💬 **Total messages**: {st.session_state.message_count}")
st.sidebar.write(f"🆔 **Conversation ID**: {st.session_state.conversation_id}")

# Load PAT
def load_pat():
    pat = st.session_state.api_key or os.environ.get('CLARIFAI_PAT')
    if not pat:
        st.error("CLARIFAI_PAT not found. Please set your API Key in the sidebar.")
        st.stop()
    return pat

pat = load_pat()

# Get LLM model
@lru_cache
def get_llm(model_id):
    try:
        model_info = models_map[model_id]
        return Clarifai(
            pat=pat,
            user_id=model_info['author'],
            app_id=model_info['app'],
            model_id=model_id
        )
    except KeyError:
        st.error(f"Model {model_id} not found in models map.")
        st.stop()

# Initialize or update the conversation chain with the chosen LLM
def initialize_conversation_chain():
    if st.session_state.chosen_llm:
        llm = get_llm(st.session_state.chosen_llm)
    else:
        llm = Clarifai(
            pat=pat,
            user_id="openai",
            app_id="chat-completion",
            model_id=default_llm
        )
    template = """
    You are an advanced AI Assistant developed to assist users with various tasks, including code generation, data analysis, and answering complex questions.

    Current conversation:
    {chat_history}
    Human: {input}
    AI Assistant:"""
    prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])

    if st.session_state.conversation is None:
        st.session_state.conversation = ConversationChain(
            prompt=prompt,
            llm=llm,
            verbose=True,
            memory=st.session_state.memory,
        )
    else:
        st.session_state.conversation.llm = llm

initialize_conversation_chain()

# Functions for code execution and processing
def execute_code(code, language):
    if language in ['html', 'css', 'javascript']:
        return execute_web_code(code, language)
    elif language in ['python', 'bash']:
        return execute_script(code, language)
    else:
        return f"Execution not supported for {language}"

def execute_web_code(code, language):
    if language == 'html':
        return code
    elif language == 'css':
        return f'<style>{code}</style>'
    elif language == 'javascript':
        return f'<script>{code}</script>'

def execute_script(code, language):
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if language == 'python':
            process = asyncio.create_subprocess_exec(
                'python', temp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        elif language == 'bash':
            process = asyncio.create_subprocess_exec(
                'bash', temp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        else:
            return f"Execution not supported for {language}"
        result = loop.run_until_complete(process)
        stdout, stderr = loop.run_until_complete(result.communicate())
        output = stdout.decode() if result.returncode == 0 else f"Error: {stderr.decode()}"
    except Exception as e:
        output = f"Execution failed: {e}"
    finally:
        os.remove(temp_file_path)
        loop.close()

    return output

def process_code_blocks(text):
    code_block_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
    code_blocks = []

    def replace_code_block(match):
        language = match.group(1) or 'text'
        code = match.group(2)
        idx = len(code_blocks)
        code_blocks.append((language.lower(), code))
        return f"<code_block_placeholder_{idx}>"

    text = code_block_pattern.sub(replace_code_block, text)

    for idx, (language, code) in enumerate(code_blocks):
        if st.session_state.code_execution_enabled:
            result = execute_code(code, language)
            with st.expander(f"Code Block {idx+1} [{language}]", expanded=False):
                st.code(code, language=language)
                if language in ['html', 'css', 'javascript']:
                    components.html(result, height=400, scrolling=True)
                else:
                    st.text("Execution Result:")
                    st.text(result)
            text = text.replace(f"<code_block_placeholder_{idx}>", "")
        else:
            text = text.replace(f"<code_block_placeholder_{idx}>", f"```{language}\n{code}\n```")

    return text

# Display chat history
def show_chat_history():
    if not st.session_state.chat_history:
        st.session_state.chat_history = load_chat_history_from_db(st.session_state.conversation_id)

    chat_list = []
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
            chat_list.append(HumanMessage(content=message["content"]))
        else:
            with st.chat_message("assistant"):
                processed_response = process_code_blocks(message["content"])
                st.markdown(processed_response, unsafe_allow_html=True)
            chat_list.append(AIMessage(content=message["content"]))

    st.session_state.conversation.memory.chat_memory = ChatMessageHistory(messages=chat_list)

# Chatbot logic
def chatbot():
    show_chat_history()
    if message := st.chat_input("Type your message here..."):
        st.session_state.message_count += 1
        st.session_state.chat_history.append({"role": "user", "content": message})
        save_message_to_db(st.session_state.conversation_id, "user", message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.predict(input=message)
                    if st.session_state.chosen_llm and 'lama' in st.session_state.chosen_llm.lower():
                        response = response.split('Human:', 1)[0]

                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.message_count += 1
                    save_message_to_db(st.session_state.conversation_id, "assistant", response)

                    processed_response = process_code_blocks(response)
                    st.markdown(processed_response, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        st.experimental_rerun()

# Main chat interface
st.title("💬 Clarifai Chat Interface")

# Add animated gradient text effect
st.markdown("""
<style>
@keyframes gradientText {
    0% { background-position: 0%; }
    100% { background-position: 100%; }
}
h1 {
    background: linear-gradient(45deg, #f3ec78, #af4261, #3498db);
    background-size: 600% 600%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientText 10s ease infinite;
}
</style>
""", unsafe_allow_html=True)

# Run the chatbot
chatbot()
