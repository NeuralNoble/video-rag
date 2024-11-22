import streamlit as st
from src.video_processor import VideoProcessor
from src.chunk_processor import ChunkProcessor, ChunkConfig
from src.vector_store import PineconeManager
from src.rag_engine import RAGEngine
from src.utils import extract_video_id
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Initialize processors
@st.cache_resource
def init_processors():
    video_processor = VideoProcessor()
    chunk_processor = ChunkProcessor()
    pinecone_manager = PineconeManager()
    return video_processor, chunk_processor, pinecone_manager


# Custom CSS
def load_css():
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        .chat-message.user {
            background-color: #EEE;
        }
        .chat-message.assistant {
            background-color: #F0F7FF;
        }
        .chat-message .avatar {
            width: 40px;
            height: 40px;
            margin-right: 1rem;
        }
        .chat-message .message {
            flex-grow: 1;
        }
        .timestamp-link {
            color: #0066cc;
            text-decoration: none;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            background-color: #e6f3ff;
            margin-right: 0.5rem;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        .stButton>button {
            border-radius: 20px;
            width: 100%;
        }
        .stTextInput>div>div>input {
            border-radius: 20px;
        }
        </style>
    """, unsafe_allow_html=True)


# Bot avatar SVG
BOT_AVATAR = """
<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="40" height="40" rx="20" fill="#1B72E8"/>
    <path d="M20 12C15.6 12 12 15.6 12 20C12 24.4 15.6 28 20 28C24.4 28 28 24.4 28 20C28 15.6 24.4 12 20 12ZM20 14.4C21.8 14.4 23.2 15.8 23.2 17.6C23.2 19.4 21.8 20.8 20 20.8C18.2 20.8 16.8 19.4 16.8 17.6C16.8 15.8 18.2 14.4 20 14.4ZM20 26C18.1 26 16.4 25.1 15.2 23.7C15.2 22.1 18.4 21.2 20 21.2C21.6 21.2 24.8 22.1 24.8 23.7C23.6 25.1 21.9 26 20 26Z" fill="white"/>
</svg>
"""

# User avatar SVG
USER_AVATAR = """
<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="40" height="40" rx="20" fill="#CCCCCC"/>
    <path d="M20 12C15.6 12 12 15.6 12 20C12 24.4 15.6 28 20 28C24.4 28 28 24.4 28 20C28 15.6 24.4 12 20 12ZM20 14.4C21.8 14.4 23.2 15.8 23.2 17.6C23.2 19.4 21.8 20.8 20 20.8C18.2 20.8 16.8 19.4 16.8 17.6C16.8 15.8 18.2 14.4 20 14.4ZM20 26C18.1 26 16.4 25.1 15.2 23.7C15.2 22.1 18.4 21.2 20 21.2C21.6 21.2 24.8 22.1 24.8 23.7C23.6 25.1 21.9 26 20 26Z" fill="white"/>
</svg>
"""


def display_chat_message(content, is_user=False):
    avatar = USER_AVATAR if is_user else BOT_AVATAR
    align = "flex-end" if is_user else "flex-start"
    avatar_div = f'<div class="avatar" style="order: {1 if is_user else 0}">{avatar}</div>'
    message_div = f'<div class="message" style="order: {0 if is_user else 1}">{content}</div>'

    st.markdown(
        f'<div class="chat-message {("user" if is_user else "assistant")}" '
        f'style="justify-content: {align}">'
        f'{message_div}{avatar_div}</div>',
        unsafe_allow_html=True
    )


def process_video(url, video_processor, chunk_processor, pinecone_manager):
    try:
        video_id = extract_video_id(url)

        # Check if video is already indexed
        if not pinecone_manager.check_video_exists(video_id):
            with st.status("Processing video...", expanded=True) as status:
                # Process video
                status.write("Downloading and transcribing video...")
                transcription, video_info, transcript_path = video_processor.process_video(url)

                # Create chunks
                status.write("Creating chunks...")
                segments = chunk_processor.read_transcript(transcript_path)
                chunks = chunk_processor.create_chunks(segments, video_id)

                # Generate embeddings
                status.write("Generating embeddings...")
                chunks_with_embeddings = chunk_processor.generate_embeddings(chunks)

                # Index chunks
                status.write("Indexing chunks...")
                pinecone_manager.index_video_chunks(chunks_with_embeddings, video_id)
                status.update(label="Video processed successfully!", state="complete")

        return True

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return False


def main():
    # Page config
    st.set_page_config(page_title="Video Chat & Search", layout="wide")
    load_css()

    # Initialize processors
    video_processor, chunk_processor, pinecone_manager = init_processors()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    # Sidebar
    with st.sidebar:
        st.title("🎥 Video Chat & Search")
        video_url = st.text_input("Enter YouTube URL:")

        if video_url:
            if process_video(video_url, video_processor, chunk_processor, pinecone_manager):
                st.session_state.rag_engine = RAGEngine(video_url)
                st.success("Video ready for chat and search!")

    # Main area
    if not video_url:
        st.info("👈 Please enter a YouTube URL in the sidebar to get started!")
        return

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["💬 Chat with Video", "🔍 Search Moments"])

    # Chat Tab
    with tab1:
        if st.session_state.rag_engine:
            # Display chat messages
            for message in st.session_state.messages:
                display_chat_message(message["content"], message["role"] == "user")

            # Chat input
            prompt = st.text_input("Ask about the video...", key="chat_input")
            send_button = st.button("Send")

            if send_button and prompt:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                display_chat_message(prompt, True)

                # Get response
                response = st.session_state.rag_engine.chat(prompt)

                # Format response with sources
                response_content = response["answer"] + "\n\n"
                if response["sources"]:
                    response_content += "📍 **Relevant moments:**\n"
                    for source in response["sources"]:
                        response_content += f'<a href="{source["url"]}" class="timestamp-link" target="_blank">{source["timestamp"]}</a>'

                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                display_chat_message(response_content, False)

                # Clear the input
                st.session_state.chat_input = ""
                st.experimental_rerun()

    # Search Tab
    with tab2:
        if st.session_state.rag_engine:
            search_query = st.text_input("Search for specific moments in the video:")
            search_button = st.button("Search")

            if search_button and search_query:
                chunks = st.session_state.rag_engine.get_relevant_chunks(search_query)

                if chunks:
                    st.write("🎯 Found these relevant moments:")
                    for chunk in chunks:
                        with st.expander(f"{chunk['start_time']}s - {chunk['end_time']}s"):
                            st.write(chunk['text'])
                            st.markdown(f'<a href="{chunk["youtube_url"]}" target="_blank">Watch this segment</a>',
                                        unsafe_allow_html=True)
                else:
                    st.info("No relevant moments found for your search query.")


if __name__ == "__main__":
    main()