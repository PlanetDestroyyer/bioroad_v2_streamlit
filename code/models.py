import streamlit as st
from ultralytics import YOLO
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import os
from config import logger 
from config import LEAF_COUNTER_MODEL , BANANA_DISEASE_MODEL , BANANA_MODEL , BANANA_STAGE_MODEL

@st.cache_resource
def load_models():
    """Load YOLO model and FAISS database with comprehensive error handling"""
    model, embeddings, db_faiss = None, None, None 
    
    try:
        
        try:
            model = YOLO('models/yolov8n.pt')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            st.warning("⚠️ Object detection model failed to load. Plant detection may be limited.")
        
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            st.warning("⚠️ Embeddings model failed to load.")
        
        try:
            if embeddings:
                db_faiss = FAISS.load_local( 
                    "banana_faiss_index", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS database loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FAISS database: {e}")
            st.warning("⚠️ Knowledge database failed to load. Responses may be limited.")
        
        return model, embeddings, db_faiss 
        
    except Exception as e:
        logger.error(f"Critical error in load_models: {e}")
        st.error(f"Critical error loading AI models: {e}")
        return None, None, None

@st.cache_resource
def load_llm(_db): 
    """Load Google Gemini LLM and RetrievalQA chain with error handling"""
    try:
        if not os.getenv('GOOGLE_API_KEY'):
            logger.error("Google API key not available")
            return None
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.1,
            max_retries=3
        )
        logger.info("LLM loaded successfully")

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and friendly banana farming expert.
            Use the following context to answer the user's question about banana plants.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Provide clear, actionable advice in plain text without any asterisks (*), bold formatting, or special characters.
            Context: {context}"""),
            ("human", "{question}") 
        ])
        logger.info("RAG Prompt template created successfully")

        
        if _db: 
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=_db.as_retriever(), 
                return_source_documents=False, 
                chain_type_kwargs={"prompt": rag_prompt} 
            )
            logger.info("RetrievalQA chain initialized successfully")
            return qa_chain
        else:
            logger.warning("FAISS database not loaded, returning direct LLM without RAG.")
            return ChatPromptTemplate.from_messages([
                ("system", """You are a helpful and friendly banana farming expert. Provide clear, actionable advice without using any asterisks (*), bold formatting, or special characters in your response. Use simple plain text only."""),
                ("human", "{query}")
            ]) | llm
            
    except Exception as e:
        logger.error(f"Error initializing LLM or RAG chain: {e}")
        st.error(f"AI service initialization failed: {e}")
        return None

if __name__ == "__main__":
    pass