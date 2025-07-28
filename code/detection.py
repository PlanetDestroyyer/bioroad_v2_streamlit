import cv2
import numpy as np
from PIL import Image
import io
import os
from config import logger
from ultralytics import YOLO
import streamlit as st

def estimate_stage(banana_present, flower_present, seedling_present, age=None):
    """
    Estimate plant growth stage, prioritizing age if provided, with error handling.
    
    Args:
        banana_present (bool): Whether bananas were detected.
        flower_present (bool): Whether flowers were detected.
        seedling_present (bool): Whether seedlings were detected.
        age (str): Plant age as a string (e.g., '1 month').
    
    Returns:
        str: Estimated growth stage.
    """
    try:
        # Parse age
        from utils import parse_age
        age_months, age_valid = parse_age(age) if age else (None, False)
        
        # Define growth stages based on age (in months)
        stages = {
            "Seeding Stage": (0, 3),
            "Vegetative Stage": (3, 6),
            "Flowering Stage": (6, 9),
            "Early Fruit Stage": (9, 12),
            "Fruit Development Stage": (12, float("inf"))
        }
        
        # Prioritize age if valid
        if age_valid:
            for stage, (min_months, max_months) in stages.items():
                if min_months <= age_months < max_months:
                    # Refine stage based on image detection, but stick to main stages
                    if stage == "Seeding Stage" and not seedling_present and age_months > 2:
                        return "Vegetative Stage"  # Move to next stage if near boundary
                    elif stage == "Flowering Stage" and not flower_present and age_months < 7:
                        return "Vegetative Stage"
                    elif stage == "Early Fruit Stage" and not banana_present and age_months < 10:
                        return "Flowering Stage"
                    elif stage == "Fruit Development Stage" and not banana_present and age_months < 14:
                        return "Early Fruit Stage"
                    return stage
            logger.info(f"Stage determined by age ({age_months:.2f} months): {stage}")
        
        # Fallback to image-based detection
        if seedling_present:
            logger.info("Stage determined by image: Seeding Stage")
            return "Seeding Stage"
        elif banana_present and flower_present:
            logger.info("Stage determined by image: Fruit Development Stage")
            return "Fruit Development Stage"
        elif flower_present:
            logger.info("Stage determined by image: Flowering Stage")
            return "Flowering Stage"
        elif banana_present:
            logger.info("Stage determined by image: Early Fruit Stage")
            return "Early Fruit Stage"
        else:
            logger.info("Stage determined by image: Vegetative Stage")
            return "Vegetative Stage"
            
    except Exception as e:
        logger.error(f"Error in estimate_stage: {e}")
        return "Unknown Stage"


def validate_plant_image(image_input, confidence_threshold=0.15):
    """
    Validate if the uploaded image contains plants/trees/vegetation
    Args:
        image_input: Can be file path (str), bytes, or PIL Image
        confidence_threshold: Minimum confidence for plant detection
    Returns: (is_plant, confidence_score, message)
    """
    try:
        # Convert input to PIL Image
        if isinstance(image_input, str):
            # File path
            if not os.path.exists(image_input):
                return False, 0.0, "File not found"
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            # Bytes data
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif hasattr(image_input, 'read'):
            # File-like object (Streamlit uploaded file)
            img = Image.open(image_input).convert("RGB")
        else:
            # Assume it's already a PIL Image
            img = image_input.convert("RGB")
        
        # Convert to OpenCV format for analysis
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Check for green dominance (vegetation indicator)
        green_ratio = check_green_dominance(img_cv)
        
        # Check image properties
        height, width = img_cv.shape[:2]
        if width < 100 or height < 100:
            return False, 0.0, "Image resolution too low (minimum 100x100)"
        
        # Basic vegetation check using color analysis
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Multiple green ranges for better detection
        green_ranges = [
            ([35, 40, 40], [85, 255, 255]),    # Primary green
            ([25, 25, 25], [95, 255, 255]),    # Extended green
            ([40, 50, 50], [80, 255, 200]),    # Leaf green
        ]
        
        total_green_pixels = 0
        total_pixels = width * height
        
        for lower, upper in green_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            total_green_pixels += np.sum(mask > 0)
        
        # Avoid double counting
        total_green_pixels = min(total_green_pixels, total_pixels)
        vegetation_ratio = total_green_pixels / total_pixels
        
        # Check for plant-like textures using edge detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / total_pixels
        
        # Combined scoring
        plant_score = (vegetation_ratio * 0.7) + (edge_ratio * 0.3)
        
        if vegetation_ratio > confidence_threshold or plant_score > confidence_threshold:
            confidence = max(vegetation_ratio, plant_score)
            return True, confidence, f"Plant detected (vegetation: {vegetation_ratio:.2%}, score: {plant_score:.2%})"
        else:
            return False, plant_score, f"No vegetation detected. Please upload a plant/leaf image. (vegetation: {vegetation_ratio:.2%})"
            
    except Exception as e:
        logger.error(f"Error in plant validation: {e}")
        return False, 0.0, f"Validation error: {str(e)}"

def check_green_dominance(image):
    """Check if image has dominant green colors"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(green_mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        return green_pixels / total_pixels
    except Exception as e:
        logger.error(f"Error in green dominance check: {e}")
        return 0.0

# Fixed detection functions
def detect_banana(image_input, yolo_model, confidence_threshold=0.5):
    """
    Fixed banana detection that works with Streamlit
    Args:
        image_input: Can be file path, bytes, or Streamlit uploaded file
        yolo_model: YOLO model instance
        confidence_threshold: Detection confidence threshold
    """
    try:
        if yolo_model is None:
            logger.warning("YOLO model not available for banana detection")
            return False
        
        # Convert input to PIL Image consistently
        if isinstance(image_input, str):
            # File path
            if not os.path.exists(image_input):
                logger.error(f"Image file not found: {image_input}")
                return False
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            # Bytes data
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif hasattr(image_input, 'read'):
            # Streamlit uploaded file
            image_input.seek(0)  # Reset file pointer
            img = Image.open(image_input).convert("RGB")
        else:
            logger.error(f"Unsupported image input type: {type(image_input)}")
            return False
        
        # Ensure image is valid
        if img.size[0] < 50 or img.size[1] < 50:
            logger.error("Image too small for detection")
            return False
        
        # Run YOLO detection with proper error handling
        try:
            results = yolo_model(img, conf=confidence_threshold, verbose=False)
            
            if not results:
                logger.info("No detection results returned")
                return False
            
            # Check each result
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        try:
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Get class name safely
                            if hasattr(result, 'names') and cls_id in result.names:
                                class_name = result.names[cls_id].lower()
                                logger.info(f"Detected: {class_name} (confidence: {confidence:.2f})")
                                
                                # Check for banana-related classes
                                if any(keyword in class_name for keyword in ['banana', 'fruit', 'plantain']):
                                    logger.info(f"Banana detected: {class_name} with confidence {confidence:.2f}")
                                    return True
                            else:
                                logger.warning(f"Unknown class ID: {cls_id}")
                        except Exception as e:
                            logger.error(f"Error processing detection box: {e}")
                            continue
            
            logger.info("No banana detected in image")
            return False
            
        except Exception as e:
            logger.error(f"Error running YOLO detection: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in detect_banana_fixed: {e}")
        return False

def detect_flower(image_input):
    """
    Fixed flower detection using color-based approach
    """
    try:
        # Convert input to OpenCV format
        if isinstance(image_input, str):
            img_cv = cv2.imread(image_input)
            if img_cv is None:
                logger.error(f"Could not read image: {image_input}")
                return False
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif hasattr(image_input, 'read'):
            # Streamlit uploaded file
            image_input.seek(0)
            image_bytes = image_input.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            logger.error(f"Unsupported input type for flower detection: {type(image_input)}")
            return False
        
        if img_cv is None:
            logger.error("Could not decode image for flower detection")
            return False
        
        # Enhanced flower detection using multiple color ranges
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Multiple flower color ranges
        flower_ranges = [
            ([0, 100, 70], [10, 255, 255]),      # Red flowers
            ([160, 100, 70], [180, 255, 255]),   # Red flowers (wrap around)
            ([120, 50, 50], [150, 255, 255]),    # Purple/violet flowers
            ([15, 100, 100], [35, 255, 255]),    # Orange/yellow flowers
            ([40, 50, 100], [60, 255, 255]),     # Yellow-green flowers
        ]
        
        total_flower_pixels = 0
        total_pixels = img_cv.shape[0] * img_cv.shape[1]
        
        for lower, upper in flower_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours for this color range
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area for flower
                    # Check if contour has flower-like properties
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if 0.3 < circularity < 1.2:  # Flower-like shape
                            total_flower_pixels += area
        
        flower_ratio = total_flower_pixels / total_pixels
        
        # Detection threshold
        if flower_ratio > 0.005:  # At least 0.5% of image should be flower-colored
            logger.info(f"Flower detected with ratio: {flower_ratio:.4f}")
            return True
        
        logger.info(f"No flower detected (ratio: {flower_ratio:.4f})")
        return False
        
    except Exception as e:
        logger.error(f"Error in flower detection: {e}")
        return False

def detect_seedling(image_input, seedling_model):
    """
    Fixed seedling detection
    """
    try:
        if seedling_model is None:
            logger.warning("Seedling model not available")
            return False
        
        # Convert input to PIL Image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.error(f"Seedling image file not found: {image_input}")
                return False
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif hasattr(image_input, 'read'):
            image_input.seek(0)
            img = Image.open(image_input).convert("RGB")
        else:
            logger.error(f"Unsupported input type for seedling detection: {type(image_input)}")
            return False
        
        try:
            results = seedling_model(img, verbose=False)
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        try:
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            if hasattr(result, 'names') and cls_id in result.names:
                                class_name = result.names[cls_id].lower()
                                logger.info(f"Seedling detection: {class_name} (confidence: {confidence:.2f})")
                                
                                if 'seedling' in class_name or 'sprout' in class_name:
                                    logger.info(f"Seedling detected: {class_name}")
                                    return True
                        except Exception as e:
                            logger.error(f"Error processing seedling detection: {e}")
                            continue
            
            logger.info("No seedling detected")
            return False
            
        except Exception as e:
            logger.error(f"Error running seedling model: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in seedling detection: {e}")
        return False