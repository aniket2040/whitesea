import streamlit as st
from llava_chatbot import LLaVAChatBot
from PIL import Image
import os

def main():
    st.title("LLaVA Chatbot")
    st.write("Upload an image and ask a question about it.")

    # Initialize the chatbot
    chatbot = LLaVAChatBot(device_map='cpu')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        st.image(Image.open(temp_file), caption="Uploaded Image")

        # Input for the question
        question = st.text_input("Ask a question about the image")

        if st.button("Get Answer"):
            try:
                answer = chatbot.start_new_chat(temp_file, question)
                st.write(answer)
                st.session_state.conv = chatbot.conv  # Store the conversation state
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Section to continue the chat
        if 'conv' in st.session_state:
            next_question = st.text_input("Ask another question")
            if st.button("Get Next Answer"):
                try:
                    answer = chatbot.continue_chat(next_question)
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
