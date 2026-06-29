# YouTube Transcript RAG

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about any YouTube video using its transcript. The application extracts the transcript, converts it into embeddings, stores them in a FAISS vector database, retrieves the most relevant transcript chunks, and generates answers using OpenAI's language models.

---

## Features

- Extract transcripts directly from YouTube videos
- Automatically split transcripts into semantic chunks
- Generate vector embeddings using OpenAI Embeddings
- Store embeddings in a FAISS vector database
- Retrieve the most relevant transcript sections for a query
- Generate context-aware answers using GPT-4o Mini
- Cache transcript indexing for faster repeated queries
- Simple Streamlit-based user interface

---

## Tech Stack

- Python
- Streamlit
- OpenAI GPT-4o Mini
- OpenAI Embeddings
- LangChain
- FAISS
- YouTube Transcript API
- python-dotenv

---

## Project Structure

```
.
├── app.py
└── README.md
```

---

## Workflow

1. Enter a YouTube video URL.
2. Extract the video's unique ID.
3. Download the English transcript using the YouTube Transcript API.
4. Combine the transcript into a single text document.
5. Split the transcript into overlapping chunks using LangChain's Recursive Character Text Splitter.
6. Generate embeddings for each chunk using OpenAI Embeddings.
7. Store the embeddings in a FAISS vector database.
8. Retrieve the most relevant transcript chunks for the user's question.
9. Build a Retrieval-Augmented Generation (RAG) pipeline using LangChain.
10. Generate an answer using GPT-4o Mini based solely on the retrieved transcript context.

---

## Libraries Used

| Library | Purpose |
|----------|---------|
| Streamlit | Interactive web application |
| LangChain | RAG pipeline orchestration |
| OpenAI | LLM inference and embeddings |
| FAISS | Vector similarity search |
| YouTube Transcript API | Transcript extraction |
| python-dotenv | Environment variable management |

---

## Use Cases

- Ask questions about long YouTube videos
- Educational video summarization
- Lecture and tutorial Q&A
- Interview transcript search
- Podcast analysis
- Content exploration without watching the full video
- Semantic search over video transcripts

---

## License

This project is intended for educational and research purposes.
