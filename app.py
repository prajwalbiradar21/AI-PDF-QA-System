import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# ======================================
# Streamlit Page Configuration
# ======================================

st.set_page_config(
    page_title="AI PDF Q&A System",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI PDF Question Answering System")
st.markdown(
    "Upload a PDF and ask questions about its contents using **LangChain**, **FAISS**, and **Llama 3.1**."
)

# ======================================
# Upload PDF
# ======================================

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

# ======================================
# Process Uploaded PDF
# ======================================

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    st.write("### File Details")
    st.write(f"**File Name:** {uploaded_file.name}")
    st.write(f"**File Type:** {uploaded_file.type}")
    st.write(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")

    # ----------------------------------
    # Save Uploaded PDF
    # ----------------------------------

    temp_path = "temp_uploaded.pdf"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())


    # ----------------------------------
    # Load PDF
    # ----------------------------------

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # ----------------------------------
    # Split into Chunks
    # ----------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    
    # ----------------------------------
    # Create Embeddings
    # ----------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ----------------------------------
    # Create FAISS Vector Database
    # ----------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # ----------------------------------
    # Create Retriever
    # ----------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    # ----------------------------------
    # Load LLM
    # ----------------------------------

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # ----------------------------------
    # Ask Question
    # ----------------------------------


    st.markdown("## 💬 Ask a question about the uploaded PDF")

    question = st.text_input(
        "",
        placeholder="Example: What are the candidate's name?"
        )
    

    if question:

        with st.spinner("Searching the document..."):

            # Retrieve Relevant Chunks
            relevant_docs = retriever.invoke(question)

            # Combine Context
            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )

            # Prompt
            prompt = f"""
            You are an AI assistant that answers questions ONLY from the uploaded PDF.
            
            Rules:
            - Use ONLY the provided context.
            - Never guess or infer information.
            - If counting items, count only the items explicitly present in the context.
            - If the information is incomplete, say so.
            - If the answer is not found, reply:
             "I couldn't find that information in the uploaded PDF."
             
             Context:
             {context}
             
             Question:
             {question}
             
             Answer:
             """
            

            # Generate Answer
            with st.spinner(" Generating answer..."):
                response = llm.invoke(prompt)

        # ----------------------------------
        # Display Answer
        # ----------------------------------

        st.subheader("Answer")
        st.write(response.content)

        # ----------------------------------
        # Show Source Chunks
        # ----------------------------------

        with st.expander("📚 View Source References"):

            for i, doc in enumerate(relevant_docs):

                st.markdown(
                    f"**Source {i+1} (Page {doc.metadata.get('page', 0)+1})**"
                    )

                st.write(doc.page_content[:500] + "...")