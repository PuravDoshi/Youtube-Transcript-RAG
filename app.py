import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Simple YT Bot", page_icon="📺")
st.title("YouTube Video Q&A")
st.markdown("Ask a single question about a video based on its transcript.")

def get_video_id(url):
    """Extracts the unique video ID from a YouTube URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url.split("/")[-1]

def format_docs(docs):
    """Joins retrieved document chunks into one block of text for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def initialize_retriever(video_id):
    """
    ST.CACHE_RESOURCE: This ensures we only download and index the 
    video once. Without this, the app would re-index on every question.
    """
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    raw_text = " ".join(chunk["text"] for chunk in fetched.to_raw_data())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.create_documents([raw_text])
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store.as_retriever(search_kwargs={"k": 4})

video_url = st.text_input("YouTube URL:")

if video_url:
    try:
        v_id = get_video_id(video_url)
        
        with st.spinner("Analyzing transcript..."):
            retriever = initialize_retriever(v_id)
        
        st.success("Transcript ready!")

        user_query = st.text_input("Ask a question about this video:")

        if user_query:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            prompt = PromptTemplate.from_template("""
                Answer ONLY from the context provided:
                {context}
                
                Question: {question}
            """)
            
            rag_chain = (
                RunnableParallel({
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                })
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.spinner("Generating answer..."):
                response = rag_chain.invoke(user_query)
                st.write("Answer:")
                st.write(response)

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
else:
    st.info("Please enter a YouTube link to begin.")