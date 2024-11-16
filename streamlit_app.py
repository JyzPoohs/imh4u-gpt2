import streamlit as st
from transformers import pipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import HuggingFacePipeline
import streamlit.components.v1 as components
import base64

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
        st.session_state.history = [
            {"origin": "ai", "message": "Hi, I am IMH4U, what can I do for you today?"}
        ]
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        model_name = "Jyz1331/gpt2-mental-health"
        generator = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=30,
            do_sample=True,
            num_beams=4,
            temperature=0.7,
            top_p=0.7,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3
        )

        llm = HuggingFacePipeline(pipeline=generator)
        
        # Update the memory buffer to store last 3 conversations
        memory = ConversationBufferWindowMemory(k=3)
        st.session_state.conversation = ConversationChain(llm=llm, memory=memory)

# Function to clean the AI response by removing the prompt template
def clean_response(response, prompt):
    return response.replace(prompt, "").strip()

# Prompt template for consistent responses
def generate_prompt(user_input):
    template = "Provide short, empathetic and supportive responses. Human: {user_input}"
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
        
        # Clean the AI response to exclude the prompt template
        response_text = clean_response(response_text, formatted_prompt)
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

# Callback function to clear the chat history
def clear_chat():
    st.session_state.history = [
        {"origin": "ai", "message": "Hi, I am IMH4U, what can I do for you today?"}
    ]
    st.session_state.token_count = 0

# Callback function to clear the input text
def clear_input_text():
    st.session_state.human_prompt = ""

# Load custom CSS and initialize session state
load_css()
initialize_session_state()

# UI for the chatbot
st.title("IMH4U Chatbot 💫")
st.write("This is IMH4U (I Am Here For You), a mental health chatbot fine-tuned using GPT-2.")

# Chat interface
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

# Clear Chat function
def clear_chat():
    # Reset the chat history and token count in session state
    st.session_state.history = [
        {"origin": "ai", "message": "Hi, I am IMH4U, what can I do for you today?"}
    ]
    st.session_state.token_count = 0

# Clear Input Text function
def clear_input_text():
    # Clear the text input in session state
    st.session_state.human_prompt = ""

# Create two columns for buttons
col1, col2 = st.columns(2)

with col1:
    # Streamlit button for Clear Text
    st.button(
        "Clear Text",
        icon="❌", 
        use_container_width=True,
        on_click=clear_input_text,
        help="Click to clear the input text",
    )

with col2:
    # Streamlit button for Clear Chat
    st.button(
        "Clear Chat",
        icon="🗑️", 
        use_container_width=True,
        on_click=clear_chat,
        help="Click to clear the chat history",
    )

# Display the chat history
with chat_placeholder:
    for chat in st.session_state.history:
        alignment = "row-reverse" if chat["origin"] == "human" else ""
        bubble_class = "human-bubble" if chat["origin"] == "human" else "ai-bubble"
        
        # Dynamically set the icon path based on the message origin
        icon_path = "img/user.jpg" if chat["origin"] == "human" else "img/bot.png" 
        
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
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback
    )

# Display token usage and conversation memory
st.caption(f"""
Used {st.session_state.token_count} tokens \n
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
