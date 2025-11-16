# -------------------- IMPORTS --------------------
import streamlit as st
import os
from dotenv import load_dotenv

# ✅ Modular LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_chains.summarize import load_summarize_chain
from langchain_chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ✅ Google Generative AI integrations
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ✅ YouTube Loader (from `langchain-yt-dlp`)
from langchain_yt_dlp import YoutubeLoaderDL

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="📊 YouTube Video Analyzer (Gemini + LangChain)",
    page_icon="🎥",
    layout="wide"
)

# -------------------- ENVIRONMENT SETUP --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("`GOOGLE_API_KEY` not found in .env file.")
    st.stop()

# -------------------- CACHE HELPERS --------------------
@st.cache_data(show_spinner=False)
def load_transcript_from_youtube(video_url: str, languages=["en"]):
    try:
        loader = YoutubeLoaderDL.from_youtube_url(
            youtube_url=video_url,
            add_video_info=True,
            languages=languages
        )
        docs = loader.load()
        full_text = " ".join([d.page_content for d in docs])
        return full_text
    except Exception as e:
        st.error(f"Failed to load transcript: {e}")
        return None


@st.cache_data(show_spinner=False)
def generate_summary(docs, llm_model, temperature=0.3):
    """Generate a concise summary of the transcript"""
    if not docs:
        return "No transcript available."
    llm = llm_model
    prompt_template = """Write a concise, easy-to-understand summary of this YouTube video transcript:
    
    "{text}"

    ===
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
    out = chain.run(docs)
    return out


@st.cache_resource(show_spinner=False)
def build_vector_store(docs, embeddings_model):
    """Build FAISS vector store from document chunks"""
    return FAISS.from_documents(documents=docs, embedding=embeddings_model)

# -------------------- SESSION STATE --------------------
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# -------------------- UI --------------------
st.title("🎥 YouTube Video Analyzer (Google Gemini + LangChain)")
st.markdown(
    "Paste a YouTube link to extract its transcript, summarize it, and ask intelligent questions!"
)

youtube_url = st.text_input("Enter the YouTube URL:")

# -------------------- MAIN PIPELINE --------------------
if st.button("Analyze Video"):
    if youtube_url:
        st.session_state.analysis_complete = False
        st.session_state.vector_store = None

        with st.spinner("⏳ Processing video..."):
            transcript_text = load_transcript_from_youtube(youtube_url)
            if transcript_text:
                # Step 1️⃣ Split transcript into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
                chunks = splitter.split_text(transcript_text)
                docs = [Document(page_content=c) for c in chunks]

                # Step 2️⃣ Initialize Gemini models
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                # Step 3️⃣ Generate summary
                summary_text = generate_summary(docs, llm_model=llm)

                # Step 4️⃣ Build vector index
                vector_store = build_vector_store(docs, embeddings_model=embeddings)

                # Save state
                st.session_state.transcript = transcript_text
                st.session_state.summary = summary_text
                st.session_state.vector_store = vector_store
                st.session_state.analysis_complete = True
                st.success("✅ Video analysis complete!")
            else:
                st.warning("Transcript not available.")
    else:
        st.warning("Please enter a YouTube URL.")

# -------------------- RESULTS SECTION --------------------
if st.session_state.analysis_complete:
    st.divider()
    st.subheader("🧾 Video Summary")
    st.write(st.session_state.summary)

    with st.expander("📜 View Full Transcript"):
        st.text_area("", st.session_state.transcript, height=300)

# -------------------- Q&A SECTION --------------------
if st.session_state.analysis_complete and st.session_state.vector_store:
    st.divider()
    st.subheader("❓ Ask a Question About the Video")

    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("💡 Generating answer..."):
                llm_qa = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)

                qa_template = """You are a helpful assistant analyzing a YouTube video transcript.

Use only the provided context to answer accurately. 
If the answer isn't clearly mentioned, start your response with:
"This specific topic wasn’t directly explained in the video, but here’s some additional relevant information:"

---
Context:
{context}

Question:
{question}

Helpful Answer:"""
                QA_PROMPT = PromptTemplate(
                    template=qa_template,
                    input_variables=["context", "question"]
                )

                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_qa,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": QA_PROMPT}
                )

                answer = qa_chain.run(user_question)
                st.success("✅ Here's the answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")
