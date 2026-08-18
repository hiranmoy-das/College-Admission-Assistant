import sys
import os
import tempfile
import re
import streamlit as st
from dotenv import load_dotenv

# Ensure local modules inside src can be imported cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_processor import extract_text_from_pdf, get_document_chunks
# --- CHANGED: Added clear_db to the import statement ---
from vector_store import add_chunks_to_db, clear_db 
from chat_chain import get_rag_chain

# Load API keys from .env file
load_dotenv()

st.set_page_config(page_title="College Admission Assistant", page_icon="🎓")
st.title("🎓 College Admission Assistant")

# Initialize Chat History if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Document Upload
with st.sidebar:
    st.header("Admin Setup")
    uploaded_file = st.file_uploader("Upload College Brochure (PDF)", type=["pdf"])
    
    if st.button("Process Document") and uploaded_file:
        with st.spinner("Clearing old data and processing new brochure..."):
            # Save uploaded file temporarily for the processor to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_pdf_path = temp_file.name
            
            try:
                # 1. Clear previous vectors from Supabase
                clear_db()
                
                # 2. Reset the current Streamlit session chat history screen
                st.session_state.messages = []
                
                # 3. Extract, chunk, and save new document data
                text = extract_text_from_pdf(temp_pdf_path)
                chunks = get_document_chunks(text)
                add_chunks_to_db(chunks)
                st.success("Database refreshed! Ready for questions on the new brochure.")
            except Exception as e:
                st.error(f"⚠️ Supabase Error: {e}\n\nPlease update `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in your `.env` file with your real Supabase project credentials.")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

# Display chat messages from current session history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Area
if user_question := st.chat_input("Ask about fee structure, available courses, etc..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching brochure..."):
            try:
                chain = get_rag_chain()
                response = chain.invoke({"input": user_question})
                raw_answer = response["answer"]
                
                clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL)
                clean_answer = clean_answer.strip()
                
                st.markdown(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})
            except Exception as e:
                error_msg = f"⚠️ Error generating response: {e}\n\nPlease check your `GROQ_API_KEY`, `SUPABASE_URL`, and `SUPABASE_SERVICE_KEY` in `.env`."
                st.error(error_msg)