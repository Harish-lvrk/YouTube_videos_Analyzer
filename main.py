import streamlit as st
import os
import re
import yt_dlp
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_data
def download_and_clean_transcript(video_url):
    """
    Downloads, cleans, and returns the transcript text for a given YouTube URL.
    This function is cached to avoid re-downloading for the same URL.
    """
    cookie_file_path = "cookies.txt"
    if not os.path.exists(cookie_file_path):
        st.error("ERROR: `cookies.txt` file not found. Please ensure it's in the project directory.")
        return None

    try:
        # Get video title to create a safe filename
        ydl_opts_info = {'quiet': True, 'cookies': cookie_file_path}
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get('title', 'video')
            sanitized_title = re.sub(r'[\\/*?:"<>|]', "", video_title)

    except Exception as e:
        st.error(f"Could not fetch video metadata. Please check the URL. Error: {e}")
        return None

    ydl_opts_download = {
        'skip_download': True, 'writeautomaticsub': True, 'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt', 'ignoreerrors': True, 'cookies': cookie_file_path,
        'outtmpl': f"{sanitized_title}.%(ext)s", 'quiet': True,
    }

    vtt_filename = f"{sanitized_title}.en.vtt"

    try:
        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            ydl.download([video_url])

        if os.path.exists(vtt_filename):
            with open(vtt_filename, 'r', encoding='utf-8') as f:
                text = f.read()
            os.remove(vtt_filename) # Clean up file after reading

            # Cleaning logic to remove VTT metadata and timestamps
            text = re.sub(r'WEBVTT.*?\n', '', text)
            text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n', '', text)
            text = re.sub(r'</?c.*?>|</?v.*?>', '', text)
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

            lines = text.split('\n')
            unique_lines = [lines[i] for i in range(len(lines)) if i == 0 or lines[i] != lines[i-1]]
            return ' '.join(unique_lines)
        else:
            st.warning("Subtitles could not be downloaded. The video may not have English subtitles.")
            return None
    except Exception as e:
        st.error(f"An error occurred during subtitle download: {e}")
        return None

@st.cache_data
def get_summary(_docs, api_key):
    """Generates a summary from document chunks. Using _docs to leverage caching."""
    if not _docs:
        return "No transcript available to summarize."
    try:
        # Reverted to the original model name as requested
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)

        prompt_template = """Write a concise, easy-to-understand summary of this video transcript:

        "{text}"

        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        summary_output = chain.invoke(_docs)
        return summary_output.get('output_text', 'Could not generate summary.')

    except Exception as e:
        st.error(f"Summarization failed. Please check your API key and permissions. Error: {e}")
        return None

# Use cache_resource for objects that can't be easily serialized, like vector DBs
@st.cache_resource
def create_vector_store(_docs, api_key):
    """Creates a FAISS vector store from document chunks and caches it."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(documents=_docs, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# --- App Initialization ---

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Session State
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# --- Main App Interface ---

st.title("📊 Simple YouTube Video Analyzer")
st.markdown("Enter a YouTube URL to extract its transcript, generate a summary, and ask questions.")

# Prerequisite checks
if not API_KEY:
    st.error("`GOOGLE_API_KEY` not found. Please create a `.env` file with your key.")
    st.stop()
if not os.path.exists("cookies.txt"):
    st.error("`cookies.txt` not found. Please place it in the project directory.")
    st.stop()

# --- Video Analysis Section ---
with st.container(border=True):
    youtube_url = st.text_input("Enter the YouTube URL here:", key="youtube_url_input")

    if st.button("Analyze Video", key="analyze_button", type="primary"):
        if youtube_url:
            # Reset state for new analysis
            st.session_state.analysis_complete = False
            st.session_state.vector_store = None # Clear previous vector store

            with st.spinner("Analyzing video... This may take a moment."):
                transcript_text = download_and_clean_transcript(youtube_url)

                if transcript_text:
                    # Split transcript into chunks (documents) once
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                    docs = text_splitter.create_documents([transcript_text])

                    # Generate summary and vector store from the same chunks
                    summary_text = get_summary(docs, api_key=API_KEY)
                    vector_store = create_vector_store(docs, api_key=API_KEY)

                    if summary_text and vector_store:
                        st.session_state.transcript = transcript_text
                        st.session_state.summary = summary_text
                        st.session_state.vector_store = vector_store
                        st.session_state.analysis_complete = True
                        st.success("Video analysis complete!")
                # Error messages are shown by the helper functions directly
        else:
            st.warning("Please enter a YouTube URL to analyze.")

# --- Results Section (Appears after analysis) ---
if st.session_state.analysis_complete:
    st.divider()

    with st.container(border=True):
        st.subheader("Analysis Results")

        if st.session_state.summary:
            st.markdown("#### 📝 Video Summary")
            st.write(st.session_state.summary)

        with st.expander("View Full Transcript"):
            st.text_area("", st.session_state.transcript, height=300)

# --- Q&A Section (New Feature) ---
if st.session_state.analysis_complete and st.session_state.vector_store:
    st.divider()

    with st.container(border=True):
        st.subheader("❓ Ask a Question About the Video")

        user_question = st.text_input("Enter your question here:", key="user_question")

        if st.button("Get Answer", key="qa_button"):
            if user_question:
                with st.spinner("Finding answer..."):
                    try:
                        # Reverted to the original model name as requested
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=API_KEY)
                        
                        # Set up the RetrievalQA chain
                        qa_prompt_template = """You are an intelligent assistant analyzing content from a YouTube video transcript.

                                                Use the following pieces of context from the video transcript to answer the user’s question as accurately as possible.

                                                If the answer is clearly explained in the transcript, respond concisely and directly using that context.

                                                If the answer is not mentioned or only partially related to the transcript, start your response with:
                                                "This specific topic wasn’t directly explained in the video, but here’s some additional relevant information:"
                                                and then provide a clear, helpful answer using your general knowledge.

                                                Do not fabricate quotes or claim something is in the video when it isn’t.

                                                Be factual, polite, and to the point.

                                                ---
                                                Context:
                                                {context}

                                                Question:
                                                {question}

                                                Helpful Answer:
                                                """
                        QA_PROMPT = PromptTemplate(
                            template=qa_prompt_template, input_variables=["context", "question"]
                        )

                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vector_store.as_retriever(),
                            return_source_documents=False, # Set to True if you want to see the source chunks
                            chain_type_kwargs={"prompt": QA_PROMPT}
                        )

                        # Get the response
                        response = qa_chain.invoke({"query": user_question})
                        st.success("Here's the answer:")
                        st.write(response["result"])

                    except Exception as e:
                        st.error(f"An error occurred while getting the answer: {e}")
            else:
                st.warning("Please enter a question.")