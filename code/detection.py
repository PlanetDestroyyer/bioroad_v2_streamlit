import cv2
import numpy as np
from PIL import Image
import io
from config import logger # Import logger from config

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

def detect_banana(image_bytes, yolo_model):
    """Detect bananas in image with comprehensive error handling"""
    try:
        if not image_bytes:
            logger.error("No image bytes provided for banana detection")
            return False
            
        if yolo_model is None:
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
            results = yolo_model(img, verbose=False)
            
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