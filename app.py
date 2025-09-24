import streamlit as st
from rag_pipeline import get_transcript, build_vectorstore, answer_question

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
st.title("📺 YouTube Video RAG Chatbot")

# -----------------------------
# Initialize session state
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# -----------------------------
# Sidebar: Video ID input
# -----------------------------
st.sidebar.header("Step 1: Enter YouTube Video ID")
video_id_input = st.sidebar.text_input("Video ID", value="6zsFZWtG6UQ")
process_video_btn = st.sidebar.button("Process Video")

if process_video_btn:
    with st.spinner("Fetching transcript and building vectorstore..."):
        transcript_text = get_transcript(video_id_input)
        if transcript_text:
            st.session_state.vectorstore = build_vectorstore(transcript_text)
            st.session_state.video_processed = True
            st.session_state.chat_history = []  # clear old chat
            st.success("✅ Video processed successfully!")
        else:
            st.error("❌ Transcript not available for this video.")

# -----------------------------
# Main area: Chat interface
# -----------------------------
st.subheader("Step 2: Ask questions about the video")

if st.session_state.vectorstore is not None and st.session_state.video_processed:

    # Display chat history first
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["query"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    # Input box at the bottom
    query = st.chat_input("Type your question here...")

    if query:
        # Show user message immediately
        st.session_state.chat_history.append({"query": query, "answer": None})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                answer = answer_question(st.session_state.vectorstore, query)
                st.markdown(answer)

        # Save bot answer
        st.session_state.chat_history[-1]["answer"] = answer

else:
    st.info("⚠️ Please enter a video ID and process the video first.")


