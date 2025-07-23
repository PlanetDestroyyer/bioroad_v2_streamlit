import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import logging
import sys

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, # Set to INFO or DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('index_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables (optional for index creation, but good practice)
load_dotenv()

def create_faiss_index(pdf_path, index_name="banana_faiss_index"):
    """
    Processes a PDF, creates embeddings, and saves them to a FAISS index.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at: {pdf_path}")
        print(f"Error: PDF file not found at '{pdf_path}'. Please ensure the path is correct.")
        return

    logger.info(f"Starting FAISS index creation for {pdf_path}")
    print(f"Loading PDF from {pdf_path}...")

    try:
        # 1. Load the PDF document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from the PDF.")
        print(f"Loaded {len(documents)} pages from the PDF.")

        # 2. Split documents into chunks
        # Adjusted chunk_size and chunk_overlap for potentially better context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split PDF into {len(chunks)} text chunks.")
        print(f"Split PDF into {len(chunks)} text chunks.")

        # 3. Create embeddings
        print("Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("HuggingFaceEmbeddings model initialized.")
        print("Creating FAISS index from document chunks. This may take a while...")

        # 4. Create and save the FAISS index
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_name)
        logger.info(f"FAISS index saved successfully to '{index_name}'.")
        print(f"FAISS index created and saved successfully to '{index_name}'.")

    except Exception as e:
        logger.exception(f"An error occurred during FAISS index creation: {e}")
        print(f"An error occurred: {e}")
        print("Please ensure you have installed all necessary libraries:")
        print("pip install pypdf langchain langchain-community sentence-transformers faiss-cpu python-dotenv")

if __name__ == "__main__":
    # Define the path to your PDF document
    pdf_document_path = "Complete Banana Plant Life Cycle Guide.pdf" # Ensure this path is correct

    # Run the index creation
    create_faiss_index(pdf_document_path)