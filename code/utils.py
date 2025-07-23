import streamlit as st
import os
import json
import uuid
from datetime import datetime
import re
from gtts import gTTS
from googletrans import Translator
from pydub import AudioSegment
import traceback
import io
from config import logger, HISTORY_FOLDER, AUDIO_FOLDER, SUPPORTED_GOOGLE_LANGUAGES # Import from config

# Initialize Google Translator with error handling
try:
    translator = Translator()
    logger.info("Google Translator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing translator: {e}")
    translator = None

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
        if lang_code not in SUPPORTED_GOOGLE_LANGUAGES:
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
    global translator
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