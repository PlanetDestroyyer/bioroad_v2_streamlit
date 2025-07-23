import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from googletrans import LANGUAGES as GOOGLE_LANGUAGES

# Setup comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
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
    else:
        os.environ['GOOGLE_API_KEY'] = api_key
        logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")

# Language selector config
LANGUAGES = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Marathi (मराठी)": "mr",
}

BASE_DIR = Path(__file__).resolve().parent.parent # Assuming config.py is in 'code/'
STATIC_FOLDER = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_FOLDER / "uploads"
AUDIO_FOLDER = STATIC_FOLDER / "audio"
HISTORY_FOLDER = BASE_DIR / "history"

def create_directories():
    """Create necessary directories for the application"""
    try:
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
        HISTORY_FOLDER.mkdir(parents=True, exist_ok=True)
        logger.info("All directories created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False

# Supported languages for translation (using Google Translate's supported languages)
LANGUAGES = {v: k for k, v in GOOGLE_LANGUAGES.items()} # Swap key-value for display name to code
# Add specific language names if Google's mapping is not ideal for display
LANGUAGES['English'] = 'en'
LANGUAGES['Hindi'] = 'hi'
LANGUAGES['Marathi'] = 'mr'
# You can add more languages here if you specifically need them in your dropdown
# and ensure they map to correct Google Translate codes.

# Ensure GOOGLE_LANGUAGES is available for checking supported languages in utils.py
SUPPORTED_GOOGLE_LANGUAGES = list(GOOGLE_LANGUAGES.keys())

# Eleven Labs API Key (Add this part)
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')
if not ELEVEN_LABS_API_KEY:
    logger.warning("ELEVEN_LABS_API_KEY not found in environment variables")
    # You might want to add a Streamlit warning in app.py if this is critical