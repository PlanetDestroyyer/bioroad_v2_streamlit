import streamlit as st
import os
import json
import uuid
from datetime import datetime
import re
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import traceback
import io
from config import logger, HISTORY_FOLDER, AUDIO_FOLDER , LEAF_COUNTER_MODEL , BANANA_DISEASE_MODEL , BANANA_MODEL , BANANA_STAGE_MODEL
import requests
from datetime import datetime, timedelta


def comprehensive_text_cleaner(text):
    """Comprehensive text cleaning function to remove formatting while preserving language structure"""
    try:
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid text input for cleaning")
            return ""
        
        logger.debug(f"Input text for cleaning: {text}")
        
        
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # Remove *text*, **text**, ***text***
        text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)  # Remove code blocks
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove markdown links
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)  # Remove images
        text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)  # Remove underline formatting
        text = re.sub(r'~~(.*?)~~', r'\1', text)  # Remove strikethrough
        text = re.sub(r'\|.*?\|', '', text)  # Remove tables
        text = re.sub(r'^[\s-]*$', '', text, flags=re.MULTILINE)  # Remove lines with only dashes/spaces
        
        
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
        
        
        text = comprehensive_text_cleaner(text)
        
        
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
    """Generate audio from text with error handling using gTTS"""
    try:
        if not text or not isinstance(text, str):
            logger.error("Invalid text provided for TTS")
            return False

        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text.strip():
            logger.error("No valid text after cleaning for TTS")
            return False

        # Define supported languages manually (gTTS supported ones)
        SUPPORTED_GOOGLE_LANGUAGES = {
            'en', 'hi', 'mr', 'ta', 'te', 'kn', 'gu', 'bn', 'ml', 'pa', 'ur'
        }

        if lang_code not in SUPPORTED_GOOGLE_LANGUAGES:
            logger.warning(f"Language '{lang_code}' not supported by gTTS. Falling back to English.")
            lang_code = 'en'

        logger.debug(f"Generating TTS for language '{lang_code}' with text: {cleaned_text}")

        # Save temporary audio file
        temp_filepath = audio_filepath.replace(".mp3", "_temp.mp3")

        try:
            tts = gTTS(text=cleaned_text, lang=lang_code, slow=False)
            tts.save(temp_filepath)
        except Exception as e:
            logger.error(f"gTTS failed with lang={lang_code}: {e}")
            if lang_code != 'en':
                # Retry in English
                try:
                    logger.info("Retrying TTS in English.")
                    tts = gTTS(text=cleaned_text, lang='en', slow=False)
                    tts.save(temp_filepath)
                    lang_code = 'en'
                except Exception as e2:
                    logger.error(f"Fallback TTS in English failed: {e2}")
                    return False
            else:
                return False

        # Load with pydub and speed adjust
        if os.path.exists(temp_filepath):
            audio = AudioSegment.from_file(temp_filepath)
            audio_with_altered_speed = audio.speedup(playback_speed=1.1)
            audio_with_altered_speed.export(audio_filepath, format="mp3")
            os.remove(temp_filepath)
            logger.info(f"Audio generated and saved: {audio_filepath}")
            return True
        else:
            logger.error("TTS temporary audio file not found after saving.")
            return False

    except Exception as e:
        logger.error(f"Unexpected error in text_to_audio: {e}")
        logger.error(traceback.format_exc())
        return False



def translate_text(text, target_lang):
    """Translate text using deep-translator with comprehensive error handling"""
    try:
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid text input for translation")
            return ""

        if target_lang == 'en':
            logger.debug("No translation needed for English")
            return text

        clean_input = comprehensive_text_cleaner(text)
        logger.debug(f"Input text for translation: {clean_input}")

        max_chunk_size = 2000  # Keep chunks smaller for reliability
        if len(clean_input) > max_chunk_size:
            chunks = [clean_input[i:i + max_chunk_size] for i in range(0, len(clean_input), max_chunk_size)]
            translated_chunks = []

            for chunk in chunks:
                try:
                    translated_text = GoogleTranslator(source='auto', target=target_lang).translate(chunk.strip())
                    translated_chunks.append(translated_text.strip())
                except Exception as e:
                    logger.error(f"Error translating chunk: {e}")
                    translated_chunks.append(chunk.strip())  # fallback

            result = ". ".join(translated_chunks)
        else:
            try:
                result = GoogleTranslator(source='auto', target=target_lang).translate(clean_input.strip())
            except Exception as e:
                logger.error(f"Translation error: {e}")
                result = clean_input

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
    
def get_weather_forecast(location):
    # API setup
    url = "https://weather-api167.p.rapidapi.com/api/weather/forecast"
    querystring = {"place": location, "units": "metric"}
    headers = {
        "x-rapidapi-key": "132d97dc1emsh74614bee7028c43p1ce01bjsna30851e6698e",
        "x-rapidapi-host": "weather-api167.p.rapidapi.com",
        "Accept": "application/json"
    }

    # Automatically set dates
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    # Make API request
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    # Function to calculate Growing Degree Days (GDD)
    def calculate_gdd(temp_min, temp_max, base_temp=10):
        avg_temp = (temp_min + temp_max) / 2
        gdd = max(0, avg_temp - base_temp)
        return round(gdd, 2)

    # Function to estimate evapotranspiration
    def estimate_et(temp, humidity):
        et = max(0, (temp - 10) * (1 - humidity / 100) * 0.1)
        return round(et, 2)

    # Process and return results
    results = []
    if response.status_code == 200 and 'list' in data:
        for entry in data['list']:
            try:
                forecast_date = datetime.strptime(entry['dt_txt'], '%Y-%m-%d %H:%M:%S').date()
            except (ValueError, KeyError):
                return [{"error": "Invalid date format or missing dt_txt in response"}]

            if forecast_date in [yesterday, today, tomorrow]:
                temp = entry['main']['temperature'] if querystring.get('units') == 'metric' else entry['main']['temperature'] - 273.15
                temp_min = entry['main']['temperature_min'] if querystring.get('units') == 'metric' else entry['main']['temperature_min'] - 273.15
                temp_max = entry['main']['temperature_max'] if querystring.get('units') == 'metric' else entry['main']['temperature_max'] - 273.15
                feels_like = entry['main']['temperature_feels_like'] if querystring.get('units') == 'metric' else entry['main']['temperature_feels_like'] - 273.15

                gdd = calculate_gdd(temp_min, temp_max)
                et = estimate_et(temp, entry['main']['humidity'])
                frost_warning = "Yes" if temp_min <= 0 else "No"
                severe_weather = "Yes" if entry['wind']['speed'] > 10 or entry.get('rain', {}).get('amount', 0) > 10 else "No"

                weather_data = {
                    "date_time": entry['dt_txt'],
                    "temperature": round(temp, 2),
                    "feels_like": round(feels_like, 2),
                    "temp_min": round(temp_min, 2),
                    "temp_max": round(temp_max, 2),
                    "humidity": entry['main']['humidity'],
                    "precipitation": entry.get('rain', {}).get('amount', 0),
                    "wind_speed": entry['wind']['speed'],
                    "wind_direction": f"{entry['wind']['direction']} ({entry['wind']['degrees']}°)",
                    "cloud_cover": entry['clouds']['cloudiness'],
                    "frost_warning": frost_warning,
                    "gdd": gdd,
                    "evapotranspiration": et,
                    "severe_weather": severe_weather,
                    "description": entry['weather'][0]['description']
                }
                results.append(weather_data)
        return results
    else:
        return [{"error": f"Error fetching data: {data.get('message', 'Unknown error')}"}]


if __name__ == "__main__":
    pass