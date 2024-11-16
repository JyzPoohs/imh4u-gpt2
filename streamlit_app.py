import streamlit as st
from streamlit_chat import message
from transformers import pipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import streamlit.components.v1 as components
from PIL import Image
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="IMH4U Chatbot", page_icon="💫")

# Function to load CSS
def load_css():
    try:
        with open("styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Ensure 'styles.css' is in the working directory.")

# Initialize session state
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        model_name = "Jyz1331/gpt2-mental-health"
        generator = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=50,  # Limit the response to 50 tokens
            temperature=0.7,    # Add temperature to control randomness
            top_p=0.95          # Add top_p for nucleus sampling
        )
        llm = HuggingFacePipeline(pipeline=generator)
        memory = ConversationBufferWindowMemory(k=5)
        st.session_state.conversation = ConversationChain(llm=llm, memory=memory)

# Prompt template for consistent responses
def generate_prompt(user_input):
    template = "Provide empathetic and supportive responses. Human: {user_input}"
    prompt = template.format(user_input=user_input)
    if len(prompt.split()) > 50:  # Adjust threshold as needed
        st.warning("Generated prompt is too long.")
    return prompt

# Define the callback for handling user input
def on_click_callback():
    human_prompt = st.session_state.human_prompt.strip()
    if not human_prompt:
        st.warning("Please enter a message.")
        return

    try:
        # Generate the formatted prompt
        formatted_prompt = generate_prompt(human_prompt)
        
        # Wrap the prompt in a list as LangChain expects a list of prompts
        llm_response = st.session_state.conversation.llm.generate([formatted_prompt])
        
        # Extract the response from the result (it's a list of results)
        response_text = llm_response.generations[0][0].text.strip()
    except ValueError as e:
        st.error(f"Error: {e}. Retrying with simplified input.")
        try:
            llm_response = st.session_state.conversation.llm.generate([human_prompt])
            response_text = llm_response.generations[0][0].text.strip()
        except Exception as retry_error:
            st.error(f"Retry failed: {retry_error}")
            return

    # Update the chat history and token count
    st.session_state.history.append({"origin": "human", "message": human_prompt})
    st.session_state.history.append({"origin": "ai", "message": response_text})
    st.session_state.token_count += len(human_prompt.split()) + len(response_text.split())

# Load custom CSS and initialize session state
load_css()
initialize_session_state()

# UI for the chatbot
st.title("IMH4U Chatbot 💫")
st.write("This is IMH4U (I Am Here For You), a mental health chatbot fine-tuned using Llama-2.")

# Chat interface
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

import base64

with chat_placeholder:
    for chat in st.session_state.history:
        alignment = "row-reverse" if chat["origin"] == "human" else ""
        bubble_class = "human-bubble" if chat["origin"] == "human" else "ai-bubble"
        
        # Dynamically set the icon path based on the message origin
        icon_path = "img/user.jpg" if chat["origin"] == "human" else "img/bot.png"  # Adjust paths accordingly
        
        # Open the appropriate image and encode it to base64
        with open(icon_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode()

        # Use the base64-encoded image in the HTML
        div = f"""
            <div class="chat-row {alignment}">
                <img class="chat-icon" src="data:image/png;base64,{img_base64}" width=32 height=32>
                <div class="chat-bubble {bubble_class}">
                    &#8203;{chat['message']}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)

# Input form for user messages
with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="secondary", 
        on_click=on_click_callback
    )

# Display token usage and conversation memory
credit_card_placeholder.caption(f"""
Used {st.session_state.token_count} tokens \n
Debug Langchain conversation: 
{st.session_state.conversation.memory.buffer}
""")

# Add custom JavaScript for "Enter" key functionality
components.html("""
<script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Submit'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            submitButton.click();
            e.preventDefault();
        }
    });
</script>
""", height=0, width=0)
