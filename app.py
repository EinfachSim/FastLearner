import streamlit as st
from rag import MistralRAGAgentRemote
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = MistralRAGAgentRemote()
    st.session_state.assistant.clear_vector_store()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = []

if 'chat_ready' not in st.session_state:
    st.session_state.chat_ready = False

# Sidebar for file upload
with st.sidebar:
    st.header("📄 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload PDF documents to chat about"
    )
    
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.files_uploaded:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Ingest the file
                    st.session_state.assistant.ingest_file(tmp_path)
                    st.session_state.files_uploaded.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    
                    # Prepare the assistant if this is the first file
                    if not st.session_state.chat_ready:
                        st.session_state.assistant.prepare_all()
                        st.session_state.chat_ready = True
                        
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
    
    # Show uploaded files
    if st.session_state.files_uploaded:
        st.subheader("Uploaded Files:")
        for filename in st.session_state.files_uploaded:
            st.text(f"• {filename}")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    # Reset all button
    if st.button("🔄 Reset All"):
        st.session_state.assistant.clear_vector_store()
        st.session_state.messages = []
        st.session_state.files_uploaded = []
        st.session_state.chat_ready = False
        st.rerun()

# Main chat interface
st.title("💬 RAG Chat Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if not st.session_state.chat_ready:
    st.info("👈 Please upload a PDF document to start chatting!")
else:
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for token in st.session_state.assistant.respond(prompt):
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_message = "There was an error processing your request. Please try again."
                message_placeholder.markdown(error_message)
                full_response = error_message
                st.error(f"Error details: {str(e)}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})