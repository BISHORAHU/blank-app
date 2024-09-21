import streamlit as st
import os
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain_community.llms import Clarifai
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
import streamlit.components.v1 as components
import re
import subprocess
import tempfile
import base64

# Set page config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# Load CSS
with open('./styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = 0
if 'code_execution_enabled' not in st.session_state:
    st.session_state.code_execution_enabled = True

# Sidebar for settings
st.sidebar.title("Settings")

# API Key input
api_key_input = st.sidebar.text_input("Enter Clarifai API Key:", type="password", key="api_key_input")
if st.sidebar.button("Set API Key"):
    st.session_state.api_key = api_key_input
    os.environ['CLARIFAI_PAT'] = st.session_state.api_key
    st.sidebar.success("API Key set successfully!")

# Model selection
def get_default_models():
    if 'DEFAULT_MODELS' not in st.secrets:
        st.error("You need to set the default models in the secrets.")
        st.stop()
    models_list = [x.strip() for x in st.secrets.DEFAULT_MODELS.split(",")]
    models_map = {}
    select_map = {}
    for m in models_list:
        id, rem = m.split(':')
        author, app = rem.split(';')
        models_map[id] = {'author': author, 'app': app}
        select_map[f"{id} : {author}"] = id
    return models_map, select_map

models_map, select_map = get_default_models()
default_llm = "GPT-4"
llms_map = {'Select an LLM': None}
llms_map.update(select_map)

chosen_llm = st.sidebar.selectbox("Select an LLM for chatting", options=llms_map.keys())
if chosen_llm and llms_map[chosen_llm] is not None:
    st.session_state.chosen_llm = llms_map[chosen_llm]

# Code execution toggle
st.session_state.code_execution_enabled = st.sidebar.checkbox("Enable Code Execution", value=st.session_state.code_execution_enabled)

# New chat button
if st.sidebar.button("New Chat"):
    st.session_state.chat_history = []
    st.session_state.message_count = 0
    st.session_state.conversation_id += 1

# Display message count and conversation ID
st.sidebar.write(f"Total messages in this chat: {st.session_state.message_count}")
st.sidebar.write(f"Conversation ID: {st.session_state.conversation_id}")

# Load PAT
def load_pat():
    pat = st.session_state.api_key or os.environ.get('CLARIFAI_PAT')
    if not pat:
        st.error("CLARIFAI_PAT not found. Please set your API Key in the sidebar.")
        st.stop()
    return pat

pat = load_pat()

# Get LLM model
def get_llm(model_id):
    return Clarifai(
        pat=pat,
        user_id=models_map[model_id]['author'],
        app_id=models_map[model_id]['app'],
        model_id=model_id
    )

if 'chosen_llm' in st.session_state:
    cur_llm = st.session_state.chosen_llm
    st.title(f"Chatting with {cur_llm}")
    llm = get_llm(cur_llm)
else:
    llm = Clarifai(
        pat=pat,
        user_id="openai",
        app_id="chat-completion",
        model_id=default_llm
    )

# Define prompt template
template = """
Current conversation:
{chat_history}
Human: {input}
AI Assistant:"""

prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])

# Initialize conversation chain
conversation = ConversationChain(
    prompt=prompt,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant", memory_key="chat_history"),
)

def execute_code(code, language):
    if language in ['html', 'css', 'javascript']:
        return execute_web_code(code, language)
    elif language in ['python', 'bash']:
        return execute_script(code, language)
    else:
        return f"Execution not supported for {language}"

def execute_web_code(code, language):
    if language == 'html':
        return f'<iframe srcdoc="{code}" width="100%" height="200" style="border:none;"></iframe>'
    elif language == 'css':
        return f'<style>{code}</style><div>CSS applied. Check the styled elements in the chat.</div>'
    elif language == 'javascript':
        js_code = f"""
        <script>
        {code}
        </script>
        <div id="js-output"></div>
        <script>
        try {{
            const output = eval(`{code}`);
            document.getElementById('js-output').innerText = output;
        }} catch (error) {{
            document.getElementById('js-output').innerText = 'Error: ' + error.message;
        }}
        </script>
        """
        return js_code

def execute_script(code, language):
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        if language == 'python':
            result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=10)
        elif language == 'bash':
            result = subprocess.run(['bash', temp_file_path], capture_output=True, text=True, timeout=10)
        
        output = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        output = "Execution timed out after 10 seconds"
    finally:
        os.remove(temp_file_path)
    
    return output

def process_code_blocks(text):
    code_block_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
    
    def replace_code_block(match):
        language = match.group(1) or 'text'
        code = match.group(2)
        if st.session_state.code_execution_enabled:
            result = execute_code(code, language.lower())
            return f"```{language}\n{code}\n```\nExecution Result:\n{result}\n"
        else:
            return f"```{language}\n{code}\n```\n"
    
    return code_block_pattern.sub(replace_code_block, text)

def show_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

def chatbot():
    if message := st.chat_input("Type your message here..."):
        st.session_state.message_count += 1
        st.session_state.chat_history.append({"role": "user", "content": message})
        show_chat_history()
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Update the LLM with the current model
                    conversation.llm = get_llm(st.session_state.chosen_llm)
                    response = conversation.predict(input=message)
                    
                    if 'chosen_llm' in st.session_state and st.session_state.chosen_llm.find('lama') > -1:
                        response = response.split('Human:', 1)[0]
                    
                    processed_response = process_code_blocks(response)
                    
                    st.markdown(processed_response, unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": processed_response})
                    st.session_state.message_count += 1
                except Exception as e:
                    st.error(f"Predict failed: {e}")

# Main chat interface
st.title("Clarifai Chat Interface")

show_chat_history()
chatbot()