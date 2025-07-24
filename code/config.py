import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
# from googletrans import LANGUAGES as GOOGLE_LANGUAGES


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


LEAF_COUNTER_MODEL = "models/yolo11x_leaf.pt"  
BANANA_MODEL = 'models/yolo8n.pt'
BANANA_DISEASE_MODEL = "models/banana_disease_prediction.h5"
BANANA_STAGE_MODEL = "models/banana_stage_classifier.h5"


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




ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')
if not ELEVEN_LABS_API_KEY:
    logger.warning("ELEVEN_LABS_API_KEY not found in environment variables")
    