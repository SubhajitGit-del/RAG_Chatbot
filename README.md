# 🎥 YouTube Video RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to:

- Fetch transcripts from YouTube videos  
- Build a vector store for semantic search  
- Ask questions about the video content and receive AI-powered contextual answers  

The chatbot is powered by **LangChain**, **Hugging Face models**, **FAISS**, and a **Streamlit UI**.

---

## 🚀 Features

### 📄 Extracts YouTube Transcripts
- Supports English transcripts via the `youtube-transcript-api`  
- Automatically fetches captions if a video has subtitles enabled  

### ✂️ Text Chunking & Embedding
- Splits transcripts into manageable chunks using `RecursiveCharacterTextSplitter`  
- Embeds text using `sentence-transformers/all-mpnet-base-v2`  

### ⚡ Efficient Semantic Search with FAISS
- Stores embeddings in a **FAISS vector store**  
- Enables fast and accurate similarity-based retrieval of transcript chunks  

### 🤖 Powered by Llama 3.1 (via Hugging Face Inference API)
- Uses **Meta’s Llama 3.1 8B Instruct model** through Hugging Face Inference API  
- No need to download large models or use GPUs locally  
- Provides **state-of-the-art text generation and reasoning**  
- Scales easily and works even in low-resource environments  

### 💻 Interactive Streamlit Web App
- Built with **Streamlit** for a clean and interactive UI  
- Enter a YouTube video ID and process transcripts in real-time  
- Ask natural language questions and receive AI-powered answers  
- Responses are displayed directly alongside the questions  

---

## 🛠️ Tech Stack
- **LangChain** – for RAG pipelines  
- **FAISS** – for vector similarity search  
- **Sentence-Transformers** – for embeddings  
- **Llama 3.1 (Hugging Face Inference API)** – for text generation  
- **Streamlit** – for interactive UI  
- **YouTube Transcript API** – for transcript extraction  

---

## 📌 Demo

![Demo Screenshot](https://github.com/SubhajitGit-del/RAG_Chatbot/blob/main/rag-yt1.png?raw=true)
![Demo Screenshot](https://github.com/SubhajitGit-del/RAG_Chatbot/blob/main/rag-yt2.png?raw=true)


---


