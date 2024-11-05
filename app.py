import streamlit as st

st.set_page_config(page_title="Summary and query", page_icon="💭")
st.write("Please upload your file, and ask me anything about it.")

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.text_input('Enter your OpenAI API key:', type='password')

# Initialize chat model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploaded_file = st.file_uploader("Choose a file", type="txt")

if uploaded_file is not None:
    # Read and split the text
    text = uploaded_file.read().decode()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Create conversation chain
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    # Display document summary
    summary_prompt = "Please provide a concise summary of the document."
    summary_response = chat([SystemMessage(content="You are a helpful assistant."), 
                           HumanMessage(content=summary_prompt + "\n\n" + text)])
    st.subheader("Document Summary:")
    st.write(summary_response.content)
    
    # Query input
    query = st.text_input("Ask a question about the document:")
    if query:
        result = st.session_state.conversation({"question": query, 
                                              "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend([(query, result["answer"])])
        
        st.subheader("Answer:")
        st.write(result["answer"])
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History:")
            for q, a in st.session_state.chat_history:
                st.write(f"Q: {q}")
                st.write(f"A: {a}")
                st.write("---")
