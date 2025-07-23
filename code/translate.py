from googletrans import Translator
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed tracing
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def trans():
    try:
        translator = Translator()
        logger.info("Google Translator initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing translator: {e}")
        translator = None
        return "Translation service unavailable"
