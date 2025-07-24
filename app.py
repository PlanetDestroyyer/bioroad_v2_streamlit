import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import uuid
from datetime import datetime
import json
from ultralytics import YOLO
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import logging
import io
import re
from gtts import gTTS
from googletrans import Translator, LANGUAGES as GOOGLE_LANGUAGES
from pydub import AudioSegment
import traceback
import sys
from pathlib import Path
from code.config import LANGUAGES, UPLOAD_FOLDER, HISTORY_FOLDER, AUDIO_FOLDER

# Setup comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed tracing
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables with error handling
try:
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found in environment variables")
        st.warning("⚠️ Google API key not configured. Some features may not work.")
    else:
        os.environ['GOOGLE_API_KEY'] = api_key
        logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    st.error(f"Configuration error: {e}")



def create_directories():
    """Create necessary directories with error handling"""
    try:
        for folder in [UPLOAD_FOLDER, HISTORY_FOLDER, AUDIO_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)
        logger.info("All directories created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        st.error(f"Setup error: Could not create required folders - {e}")
        return False

# Create directories
if not create_directories():
    st.stop()

# Initialize Google Translator with error handling
try:
    translator = Translator()
    logger.info("Google Translator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing translator: {e}")
    translator = None
    st.warning("Translation service unavailable")

# Streamlit configuration
try:
    st.set_page_config(page_title="Banana Plant Care Advisor", layout="wide")
    st.sidebar.selectbox("🔊 Select Language", options=list(LANGUAGES.keys()), key="selected_language")
except Exception as e:
    logger.error(f"Error setting up Streamlit config: {e}")

@st.cache_resource
def load_models():
    """Load YOLO model and FAISS database with comprehensive error handling"""
    model, embeddings, db = None, None, None
    
    try:
        # Load YOLO model
        try:
            model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            st.warning("⚠️ Object detection model failed to load. Plant detection may be limited.")
        
        # Load embeddings
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            st.warning("⚠️ Embeddings model failed to load.")
        
        # Load FAISS database
        try:
            if embeddings:
                db = FAISS.load_local(
                    "banana_faiss_index", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS database loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FAISS database: {e}")
            st.warning("⚠️ Knowledge database failed to load. Responses may be limited.")
        
        return model, embeddings, db
        
    except Exception as e:
        logger.error(f"Critical error in load_models: {e}")
        st.error(f"Critical error loading AI models: {e}")
        return None, None, None

model, embeddings, db = load_models()

# Enhanced prompt template
try:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and friendly banana farming expert. Provide clear, actionable advice without using any asterisks (*), bold formatting, or special characters in your response. Use simple plain text only."""),
        ("human", "{query}")
    ])
    logger.info("Prompt template created successfully")
except Exception as e:
    logger.error(f"Error creating prompt template: {e}")
    prompt = None

@st.cache_resource
def load_llm():
    """Load Google Gemini LLM with error handling"""
    try:
        if not os.getenv('GOOGLE_API_KEY'):
            logger.error("Google API key not available")
            return None
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.1,
            max_retries=3
        )
        
        if prompt:
            qa_chain = prompt | llm
            logger.info("LLM initialized successfully")
            return qa_chain
        else:
            logger.error("Prompt template not available")
            return None
            
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        st.error(f"AI service initialization failed: {e}")
        return None

qa_chain = load_llm()

def comprehensive_text_cleaner(text):
    """Comprehensive text cleaning function to remove formatting while preserving language structure"""
    try:
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid text input for cleaning")
            return ""
        
        logger.debug(f"Input text for cleaning: {text}")
        
        # Remove markdown-specific formatting
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # Remove *text*, **text**, ***text***
        text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)  # Remove code blocks
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove markdown links
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)  # Remove images
        text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)  # Remove underline formatting
        text = re.sub(r'~~(.*?)~~', r'\1', text)  # Remove strikethrough
        text = re.sub(r'\|.*?\|', '', text)  # Remove tables
        text = re.sub(r'^[\s-]*$', '', text, flags=re.MULTILINE)  # Remove lines with only dashes/spaces
        
        # Preserve Devanagari characters (Hindi, Marathi) and essential punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-\n\u0900-\u097F]', ' ', text)  # Include Devanagari range
        
        # Normalize whitespace
        text = re.sub(r'\n{2,}', '\n', text)  # Reduce multiple newlines to one
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        
        logger.debug(f"Cleaned text: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Error in comprehensive_text_cleaner: {e}")
        return str(text) if text else ""

def clean_text_for_tts(text):
    """Clean text specifically for text-to-speech while preserving sentence structure"""
    try:
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid text input for TTS cleaning")
            return ""
        
        logger.debug(f"Input text for TTS cleaning: {text}")
        
        # Apply comprehensive cleaning
        text = comprehensive_text_cleaner(text)
        
        # Preserve sentence boundaries for TTS
        text = re.sub(r'\n+', '. ', text)  # Replace newlines with period + space
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = text.strip()
        
        logger.debug(f"TTS cleaned text: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Error in clean_text_for_tts: {e}")
        return str(text) if text else ""

def clean_text_for_display(text):
    """Clean text for display with preserved structure"""
    try:
        logger.debug(f"Input text for display cleaning: {text}")
        cleaned_text = comprehensive_text_cleaner(text)
        logger.debug(f"Display cleaned text: {cleaned_text}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error in clean_text_for_display: {e}")
        return str(text) if text else ""

def text_to_audio(text, audio_filepath, lang_code='en'):
    """Generate audio from text with comprehensive error handling"""
    try:
        if not text or not isinstance(text, str):
            logger.error("Invalid text provided for TTS")
            return False
            
        cleaned_text = clean_text_for_tts(text)
        
        if not cleaned_text.strip():
            logger.error("No valid text after cleaning for TTS")
            return False
        
        # Verify language code
        if lang_code not in GOOGLE_LANGUAGES:
            logger.warning(f"Language {lang_code} not supported by gTTS, falling back to English")
            lang_code = 'en'
        
        logger.debug(f"Generating TTS for text: {cleaned_text} in language: {lang_code}")
        
        # Generate TTS
        try:
            tts = gTTS(text=cleaned_text, lang=lang_code, slow=False)
        except Exception as e:
            logger.error(f"gTTS initialization error: {e}")
            # Fallback to English
            try:
                tts = gTTS(text=cleaned_text, lang='en', slow=False)
                logger.info("Fallback to English TTS")
            except Exception as e2:
                logger.error(f"Fallback TTS failed: {e2}")
                return False
        
        # Create temporary file
        temp_filepath = audio_filepath.replace(".mp3", "_temp.mp3")
        
        try:
            tts.save(temp_filepath)
        except Exception as e:
            logger.error(f"Error saving TTS file: {e}")
            return False
        
        # Process audio with pydub
        try:
            if os.path.exists(temp_filepath):
                audio = AudioSegment.from_file(temp_filepath)
                audio_with_altered_speed = audio.speedup(playback_speed=1.1)  # Adjusted for natural human-like pace
                audio_with_altered_speed.export(audio_filepath, format="mp3")
                
                # Clean up temporary file
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    
                logger.info(f"Audio generated successfully: {audio_filepath}")
                return True
            else:
                logger.error("Temporary audio file not created")
                return False
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            except:
                pass
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in text_to_audio: {e}")
        logger.error(traceback.format_exc())
        return False

def translate_text(text, target_lang):
    """Translate text with comprehensive error handling"""
    try:
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid text input for translation")
            return ""
            
        if target_lang == 'en' or not translator:
            logger.debug("No translation needed or translator unavailable")
            return text
            
        # Clean text before translation
        clean_input = comprehensive_text_cleaner(text)
        logger.debug(f"Input text for translation: {clean_input}")
        
        # Split long text into chunks to avoid translation limits
        max_chunk_size = 2000  # Reduced for better accuracy
        if len(clean_input) > max_chunk_size:
            chunks = [clean_input[i:i+max_chunk_size] 
                     for i in range(0, len(clean_input), max_chunk_size)]
            translated_chunks = []
            
            for chunk in chunks:
                try:
                    translated = translator.translate(chunk.strip(), dest=target_lang)
                    translated_chunks.append(translated.text.strip())
                except Exception as e:
                    logger.error(f"Error translating chunk: {e}")
                    translated_chunks.append(chunk.strip())  # Use original if translation fails
                    
            # Join chunks with period and space for sentence boundaries
            result = ". ".join(translated_chunks)
        else:
            try:
                translated = translator.translate(clean_input, dest=target_lang)
                result = translated.text.strip()
            except Exception as e:
                logger.error(f"Translation error: {e}")
                result = clean_input  # Return original text if translation fails
        
        # Clean the translated result
        cleaned_result = comprehensive_text_cleaner(result)
        logger.debug(f"Translated and cleaned text: {cleaned_result}")
        return cleaned_result
        
    except Exception as e:
        logger.error(f"Unexpected error in translate_text: {e}")
        return comprehensive_text_cleaner(text) if text else ""

def get_session_id():
    """Get or create session ID with error handling"""
    try:
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    except Exception as e:
        logger.error(f"Error getting session ID: {e}")
        return str(uuid.uuid4())

def save_to_history(session_id, data):
    """Save data to history with comprehensive error handling"""
    try:
        if not session_id or not data:
            logger.error("Invalid session_id or data for history saving")
            return False
            
        history_file = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
        
        # Load existing history
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"Error reading existing history: {e}")
                history = []
        
        # Add new data
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        history.append(history_entry)
        
        # Save updated history with UTF-8 encoding
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            logger.info(f"History saved successfully for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error writing history file: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in save_to_history: {e}")
        return False

def load_history(session_id):
    """Load history with error handling"""
    try:
        if not session_id:
            return []
            
        history_file = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading history file: {e}")
                return []
        return []
        
    except Exception as e:
        logger.error(f"Error in load_history: {e}")
        return []

def estimate_stage(banana_present, flower_present):
    """Estimate plant growth stage with error handling"""
    try:
        if banana_present and flower_present:
            return "Fruit Development Stage"
        elif flower_present:
            return "Flowering Stage"
        elif banana_present:
            return "Early Fruit Stage"
        else:
            return "Vegetative Stage"
    except Exception as e:
        logger.error(f"Error in estimate_stage: {e}")
        return "Unknown Stage"

def detect_flower(image_bytes):
    """Detect flowers in image with comprehensive error handling"""
    try:
        if not image_bytes:
            logger.error("No image bytes provided for flower detection")
            return False
            
        # Open and process image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error processing image for flower detection: {e}")
            return False
        
        # HSV color detection
        try:
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for flower detection
            mask1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 100, 70]), np.array([180, 255, 255]))
            mask3 = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([150, 255, 255]))
            
            combined_mask = mask1 + mask2 + mask3
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for significant contours
            flower_detected = any(cv2.contourArea(c) > 1000 for c in contours)
            logger.info(f"Flower detection result: {flower_detected}")
            return flower_detected
            
        except Exception as e:
            logger.error(f"Error in HSV processing for flower detection: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in detect_flower: {e}")
        return False

def detect_banana(image_bytes):
    """Detect bananas in image with comprehensive error handling"""
    try:
        if not image_bytes:
            logger.error("No image bytes provided for banana detection")
            return False
            
        if model is None:
            logger.warning("YOLO model not available for banana detection")
            return False
        
        # Process image
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Error opening image for banana detection: {e}")
            return False
        
        # Run YOLO detection
        try:
            results = model(img, verbose=False)
            
            for result in results:
                if result.boxes is not None and result.boxes.cls is not None:
                    for cls in result.boxes.cls.tolist():
                        try:
                            class_name = result.names[int(cls)].lower()
                            if 'banana' in class_name:
                                logger.info("Banana detected in image")
                                return True
                        except Exception as e:
                            logger.error(f"Error processing detection result: {e}")
                            continue
            
            logger.info("No banana detected in image")
            return False
            
        except Exception as e:
            logger.error(f"Error running YOLO detection: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in detect_banana: {e}")
        return False

# Main Streamlit App
try:
    st.title("🍌 Banana Plant Care Advisor")
    st.markdown("Upload banana plant images and get expert multilingual care advice.")

    session_id = get_session_id()
    
    try:
        lang_code = LANGUAGES[st.session_state.get("selected_language", "English")]
    except Exception as e:
        logger.error(f"Error getting language code: {e}")
        lang_code = "en"

    # Create tabs with error handling
    try:
        tab1, tab2 = st.tabs(["📸 Plant Analysis", "❓ Follow-up Questions"])
    except Exception as e:
        logger.error(f"Error creating tabs: {e}")
        st.error("Interface error. Please refresh the page.")
        st.stop()

    with tab1:
        with st.expander("Upload Plant Photos & Get Advice", expanded=True):
            st.header("New Analysis")
            
            try:
                name = st.text_input("Plant Name (Optional)", key="plant_name_input")
                age = st.text_input("Plant Age (e.g., '3 months')", key="plant_age_input")
                uploaded_files = st.file_uploader(
                    "Upload Plant Images", 
                    type=["png", "jpg", "jpeg", "webp"], 
                    accept_multiple_files=True
                )
            except Exception as e:
                logger.error(f"Error creating input widgets: {e}")
                st.error("Error setting up input fields")

            if st.button("Analyze Plant"):
                try:
                    if not uploaded_files:
                        st.warning("Please upload at least one image.")
                    elif qa_chain is None:
                        st.error("AI service is not available. Please check configuration.")
                    else:
                        with st.spinner("Analyzing your plant(s)..."):
                            results = []
                            analysis_id = str(uuid.uuid4())

                            for uploaded_file in uploaded_files:
                                try:
                                    # Read file
                                    file_bytes = uploaded_file.read()
                                    filename = uploaded_file.name
                                    unique_filename = f"{analysis_id}_{filename}"
                                    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
                                    
                                    # Save file
                                    try:
                                        with open(filepath, "wb") as f:
                                            f.write(file_bytes)
                                        logger.info(f"File saved: {filepath}")
                                    except Exception as e:
                                        logger.error(f"Error saving file {filename}: {e}")
                                        continue

                                    # Detect features
                                    banana_present = detect_banana(file_bytes)
                                    flower_present = detect_flower(file_bytes)
                                    stage = estimate_stage(banana_present, flower_present)

                                    # Generate query
                                    query = f"""
                                    Analyzing banana plant named '{name}' which is {age} old.
                                    Fruits detected: {'Yes' if banana_present else 'No'}
                                    Flowers detected: {'Yes' if flower_present else 'No'}
                                    Stage: {stage}
                                    Provide clear, practical care advice in plain text without any asterisks or formatting.
                                    """

                                    # Get AI response
                                    try:
                                        response = qa_chain.invoke({"query": query})
                                        advice = response.content if hasattr(response, 'content') else str(response)
                                        advice = comprehensive_text_cleaner(advice)
                                        translated_advice = translate_text(advice, lang_code)
                                        logger.info("AI response generated successfully")
                                    except Exception as e:
                                        logger.error(f"LLM error: {e}")
                                        advice = "AI service temporarily unavailable. Please try again later."
                                        translated_advice = translate_text(advice, lang_code)

                                    results.append({
                                        "image_path": filepath,
                                        "banana_detected": banana_present,
                                        "flower_detected": flower_present,
                                        "stage": stage,
                                        "advice": advice,
                                        "translated_advice": translated_advice,
                                        "filename": filename
                                    })

                                except Exception as e:
                                    logger.error(f"Error processing file {uploaded_file.name}: {e}")
                                    st.error(f"Error processing {uploaded_file.name}: {e}")
                                    continue

                            # Save analysis
                            if results:
                                analysis_data = {
                                    "analysis_id": analysis_id,
                                    "name": name,
                                    "age": age,
                                    "results": results,
                                    "questions": []
                                }

                                if save_to_history(session_id, analysis_data):
                                    st.session_state.current_analysis = analysis_data
                                    st.rerun()
                                else:
                                    st.warning("Analysis completed but couldn't save to history")
                            else:
                                st.error("No images were successfully processed")

                except Exception as e:
                    logger.error(f"Critical error in plant analysis: {e}")
                    logger.error(traceback.format_exc())
                    st.error(f"Analysis failed: {e}")

        # Display results
        if 'current_analysis' in st.session_state:
            try:
                current_analysis = st.session_state.current_analysis
                st.subheader("Latest Analysis Results")
                st.write(f"**Plant Name:** {current_analysis.get('name', 'Unknown')}")
                st.write(f"**Plant Age:** {current_analysis.get('age', 'Unknown')}")

                for i, result in enumerate(current_analysis.get('results', [])):
                    try:
                        st.write("---")
                        
                        # Display image safely
                        try:
                            if os.path.exists(result['image_path']):
                                st.image(result['image_path'], caption=f"Image {i+1}", use_container_width=True)
                            else:
                                st.warning(f"Image {i+1} file not found")
                        except Exception as e:
                            logger.error(f"Error displaying image {i+1}: {e}")
                            st.warning(f"Could not display image {i+1}")
                        
                        st.write(f"**Banana Detected:** {'Yes' if result.get('banana_detected', False) else 'No'}")
                        st.write(f"**Flower Detected:** {'Yes' if result.get('flower_detected', False) else 'No'}")
                        st.write(f"**Estimated Stage:** {result.get('stage', 'Unknown')}")
                        
                        st.subheader("Care Advice:")
                        advice_text = result.get('translated_advice', 'No advice available')
                        st.write(advice_text)

                        # Audio generation
                        advice_audio_filepath = os.path.join(AUDIO_FOLDER, f"advice_{current_analysis['analysis_id']}_{i}.mp3")
                        if st.button(f"Play Advice for Image {i+1}", key=f"play_advice_{i}"):
                            with st.spinner("Generating audio..."):
                                try:
                                    if text_to_audio(advice_text, advice_audio_filepath, lang_code=lang_code):
                                        st.audio(advice_audio_filepath, format='audio/mp3')
                                        st.caption(f"🔈 Language: {st.session_state.get('selected_language', 'English')}")
                                    else:
                                        st.error("Audio generation failed. Please try again.")
                                except Exception as e:
                                    logger.error(f"Error in audio generation: {e}")
                                    st.error("Audio generation failed.")

                    except Exception as e:
                        logger.error(f"Error displaying result {i}: {e}")
                        st.error(f"Error displaying result {i+1}")

            except Exception as e:
                logger.error(f"Error displaying current analysis: {e}")
                st.error("Error displaying analysis results")

    with tab2:
        st.header("Follow-up Questions")
        try:
            if 'current_analysis' in st.session_state:
                question = st.text_area("Ask a follow-up question about your plant:", key="follow_up_question")
                if st.button("Submit Question"):
                    try:
                        if not question.strip():
                            st.warning("Please enter a question.")
                        elif qa_chain is None:
                            st.error("AI service is not available.")
                        else:
                            with st.spinner("Processing your question..."):
                                try:
                                    # Translate question to English if needed
                                    question_en = question if lang_code == 'en' else translate_text(question, 'en')
                                    
                                    # Get AI response
                                    response = qa_chain.invoke({"query": question_en})
                                    answer = response.content if hasattr(response, 'content') else str(response)
                                    answer = comprehensive_text_cleaner(answer)
                                    translated_answer = translate_text(answer, lang_code)

                                    question_data = {
                                        "question": question,
                                        "answer": answer,
                                        "translated_answer": translated_answer,
                                        "timestamp": datetime.now().isoformat()
                                    }

                                    # Update history
                                    current_analysis = st.session_state.current_analysis
                                    current_analysis['questions'].append(question_data)
                                    
                                    if save_to_history(session_id, current_analysis):
                                        st.session_state.current_analysis = current_analysis
                                        st.rerun()
                                    else:
                                        st.warning("Response generated but couldn't save to history")
                                        
                                except Exception as e:
                                    logger.error(f"Error processing question: {e}")
                                    st.error("Error processing your question. Please try again.")

                    except Exception as e:
                        logger.error(f"Error in question submission: {e}")
                        st.error("Error submitting question")

                # Display question history
                try:
                    if st.session_state.current_analysis.get('questions'):
                        st.subheader("Question History")
                        for i, q in enumerate(st.session_state.current_analysis['questions']):
                            try:
                                st.write(f"**Q{i+1}:** {q.get('question', 'Question not available')}")
                                answer_text = q.get('translated_answer', 'Answer not available')
                                st.write(f"**A{i+1}:** {answer_text}")
                                
                                # Audio for answers
                                question_audio_filepath = os.path.join(AUDIO_FOLDER, f"question_{current_analysis['analysis_id']}_{i}.mp3")
                                if st.button(f"Play Answer {i+1}", key=f"play_question_{i}"):
                                    with st.spinner("Generating audio..."):
                                        try:
                                            if text_to_audio(answer_text, question_audio_filepath, lang_code=lang_code):
                                                st.audio(question_audio_filepath, format='audio/mp3')
                                                st.caption(f"🔈 Language: {st.session_state.get('selected_language', 'English')}")
                                            else:
                                                st.error("Audio generation failed. Please try again.")
                                        except Exception as e:
                                            logger.error(f"Error generating question audio: {e}")
                                            st.error("Audio generation failed.")
                                st.write("---")
                            except Exception as e:
                                logger.error(f"Error displaying question {i}: {e}")
                                st.error(f"Error displaying question {i+1}")
                except Exception as e:
                    logger.error(f"Error displaying question history: {e}")
                    st.error("Error displaying question history")
            else:
                st.info("Please analyze a plant first to ask follow-up questions.")
        except Exception as e:
            logger.error(f"Error in follow-up questions tab: {e}")
            st.error("Error setting up follow-up questions")

except Exception as e:
    logger.error(f"Critical application error: {e}")
    logger.error(traceback.format_exc())
    st.error("Application encountered a critical error. Please refresh the page.")

# Footer with error handling
try:
    st.markdown("---")
    st.markdown("### 🌱 Tips for Better Results")
    st.markdown("""
    - Upload clear, well-lit images of your banana plants
    - Include multiple angles if possible
    - Provide plant age and name for more accurate advice
    - Ask specific questions about care concerns
    """)
except Exception as e:
    logger.error(f"Error displaying footer: {e}")

# Health check
try:
    if st.sidebar.button("🔧 System Status"):
        st.sidebar.write("**System Status:**")
        st.sidebar.write(f"✅ YOLO Model: {'Loaded' if model else '❌ Failed'}")
        st.sidebar.write(f"✅ Embeddings: {'Loaded' if embeddings else '❌ Failed'}")
        st.sidebar.write(f"✅ Database: {'Loaded' if db else '❌ Failed'}")
        st.sidebar.write(f"✅ LLM: {'Connected' if qa_chain else '❌ Failed'}")
        st.sidebar.write(f"✅ Translator: {'Available' if translator else '❌ Failed'}")
        st.sidebar.write(f"✅ API Key: {'Configured' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
except Exception as e:
    logger.error(f"Error in system status: {e}")