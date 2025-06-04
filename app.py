import streamlit as st
import os
import time
import base64
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="LAWGPT", layout="wide", initial_sidebar_state="collapsed")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Custom CSS styling
custom_css = """
<style>
body {
    background-color: #121212;
    color: #fff;
    font-family: 'Segoe UI', sans-serif;
}
.chat-container {
    max-width: 900px;
    margin: auto;
    padding: 1rem;
}
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
}
.header-text {
    flex: 1;
}
.header-image img {
    width: 150px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
}
.legal-disclaimer {
    background-color: #1e1e1e;
    padding: 1rem;
    border-left: 5px solid #f39c12;
    margin-top: 1rem;
    border-radius: 10px;
}
.warning-message {
    background-color: #331a00;
    color: #ffcc00;
    padding: 10px;
    margin-top: 10px;
    border-radius: 5px;
    font-weight: bold;
}
.dancing-script {
    font-family: 'Dancing Script', cursive;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Header image and title
try:
    image_path = "saul.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()
        st.markdown(f"""
            <div class="header-container">
                <div class="header-text">
                    <h1 style="color: white; font-size: 2.5rem;">
                        <span class="dancing-script">Better Call Bot!</span>
                    </h1>
                    <p style="color: #e0e0e0; font-size: 1.2rem;">
                        Did you know that you have rights? The Constitution says you do. And so do I.
                    </p>
                </div>
                <div class="header-image">
                    <img src="data:image/jpeg;base64,{encoded_image}" alt="Saul Goodman">
                </div>
            </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Unable to load image: {e}")

# Disclaimer
st.markdown("""
<div class="legal-disclaimer">
    <h4>⚠️ Legal Information Disclaimer</h4>
    <p>This chatbot provides general legal information, NOT legal advice. The information provided:</p>
    <ul>
        <li>Is for informational purposes only</li>
        <li>Is not a substitute for professional legal counsel</li>
        <li>May not be up-to-date or applicable to your jurisdiction</li>
        <li>Should not be relied upon for making legal decisions</li>
    </ul>
    <p><strong>Please consult with a qualified attorney for specific legal advice.</strong></p>
</div>
""", unsafe_allow_html=True)

# Reset conversation
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Load vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt template
prompt_template = """
<s>[INST]You are a legal information chatbot with strict limitations. Follow these guidelines:

1. NEVER provide specific legal advice
2. If the question seeks specific legal advice or involves complex legal matters, respond with a warning to seek professional legal counsel
3. Only provide publicly available legal information with proper citations
4. Use clear qualifying language (e.g., "generally," "typically," "it may depend")
5. If unsure, explicitly state the limitations of the information
6. For questions about:
   - Ongoing legal proceedings: Decline to comment
   - Specific legal strategy: Refer to an attorney
   - Complex legal interpretation: Emphasize need for professional counsel

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# LLM + QA Chain
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Chat UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render past messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role"), avatar="👤" if message.get("role") == "user" else "⚖️"):
        content = message.get("content")
        if "Sources:" in content:
            main, sources = content.split("Sources:", 1)
            st.write(main)
            st.markdown("**Sources:**" + sources)
        else:
            st.write(content)

# Risky content checker
def check_for_risky_content(response):
    risky_keywords = ['you should', 'I advise', 'you must', 'definitely', 'always', 'never']
    return any(keyword in response.lower() for keyword in risky_keywords)

# Chat input
input_prompt = st.chat_input("Ask your legal question...")

if input_prompt:
    with st.chat_message("user", avatar="👤"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant", avatar="⚖️"):
        with st.status("Analyzing your question...", expanded=True):
            result = qa.invoke(input=input_prompt)
            response_text = result["answer"]

            if check_for_risky_content(response_text):
                st.markdown("""
                    <div class="warning-message">
                        ⚠️ This response may contain general guidance. Please consult with a qualified attorney for specific advice.
                    </div>
                """, unsafe_allow_html=True)

            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_text:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")

        col1, col2, col3 = st.columns([4, 1, 4])
        with col2:
            st.button('🗑️ Clear Chat', on_click=reset_conversation, key="clear_chat", help="Clear the conversation history", type="secondary", use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

st.markdown('</div>', unsafe_allow_html=True)
