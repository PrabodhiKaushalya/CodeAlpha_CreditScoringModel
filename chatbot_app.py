import streamlit as st
from nltk.chat.util import Chat, reflections

# 1. Chatbot Logic (Same as before)
pairs = [
    [r"hi|hello|hey", ["Hello! How can I help you today?", "Hi there!", "Hey!"]],
    [r"what is your name?", ["I am your AI Assistant.",]],
    [r"how are you?", ["I'm doing great! How about you?",]],
    [r"quit", ["Bye! Take care.",]]
]

chat = Chat(pairs, reflections)

# 2. Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 CodeAlpha AI Chatbot")
st.write("Type something below to chat with me!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get chatbot response
    response = chat.respond(prompt)
    if not response:
        response = "I'm sorry, I don't understand that."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})