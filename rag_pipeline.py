# rag_pipeline.py
import os
from dotenv import load_dotenv

# Load local .env only for development. .env must be in .gitignore and not committed.
load_dotenv()

# Read Hugging Face token from environment
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    # In production (Render) the token comes from the environment.
    # For local dev you can put the token in a local .env file.
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN environment variable")

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Embedding Model (local Hugging Face)
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},        # change to "cuda" if you have GPU
    encode_kwargs={"normalize_embeddings": True},
)

# -----------------------------
# Chat Model (Hugging Face endpoint)
# -----------------------------
# Pass the token to the HF endpoint client if the constructor supports it,
# otherwise the client will read HUGGINGFACEHUB_API_TOKEN from env automatically.
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)

chat_model = ChatHuggingFace(llm=llm)


# -----------------------------
# Function: Fetch transcript from YouTube
# -----------------------------
def get_transcript(video_id: str) -> str:
    """
    Fetch transcript text for a given YouTube video ID.
    Returns full transcript as a single string.
    """
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        # Older/newer API shapes differ; handle both
        if hasattr(transcript, "snippets"):
            full_text = " ".join(snippet.text for snippet in transcript.snippets)
        elif isinstance(transcript, list):
            # transcripts returned as list of dicts
            full_text = " ".join(item.get("text", "") for item in transcript)
        else:
            full_text = str(transcript)
        return full_text
    except TranscriptsDisabled:
        return None
    except Exception:
        return None


# -----------------------------
# Function: Build FAISS Vectorstore
# -----------------------------
def build_vectorstore(full_text: str):
    """
    Split transcript into chunks, embed them, and store in FAISS.
    Returns a retriever for later querying.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([full_text])

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# -----------------------------
# Function: Answer a user query
# -----------------------------
def answer_question(retriever, query: str) -> str:
    """
    Given a retriever (vectorstore) and a query,
    build context, run through LLM, and return the answer.
    """
    # Prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # Helper to format retrieved docs
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Build RAG chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | chat_model | parser

    # Run query
    answer = main_chain.invoke(query)
    return answer


if __name__ == "__main__":
    # Test run
    video_id = "6zsFZWtG6UQ"   # pick a video with captions
    print("Fetching transcript...")
    text = get_transcript(video_id)

    if text:
        print("✅ Transcript fetched. Sample:\n", text[:300], "\n---")

        print("Building vectorstore...")
        retriever = build_vectorstore(text)

        query = "Is the topic of Market Research discussed in this video? If yes, what was discussed?"
        print("Asking:", query)

        answer = answer_question(retriever, query)
        print("🤖 Answer:\n", answer)
    else:
        print("❌ Transcript not available for this video.")
