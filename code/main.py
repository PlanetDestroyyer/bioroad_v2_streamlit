import streamlit as st
import os
import sys
import traceback
import uuid
from config import logger, create_directories, LANGUAGES, UPLOAD_FOLDER, AUDIO_FOLDER
from models import load_models, load_llm 
from utils import (
    get_session_id, save_to_history, load_history, 
    translate_text, text_to_audio, comprehensive_text_cleaner
)
from detection import detect_banana, detect_flower, estimate_stage
from config import LEAF_COUNTER_MODEL, BANANA_DISEASE_MODEL, BANANA_MODEL, BANANA_STAGE_MODEL
from datetime import datetime
from leaf_counter import analyze_leaf_colors
from banana_disease_detection import predict_banana_disease
import cv2
from ultralytics import YOLO
import base64
st.set_page_config(page_title="Home",layout="wide",initial_sidebar_state="auto",menu_items=None)  

def hideAll():
    hide = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """   
    st.markdown(hide, unsafe_allow_html=True)
hideAll()

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/{image_file.split('.')[-1]};base64,{encoded_string}"

def run_app():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({add_bg_from_local("static/styles/bg/background_image_4.jpg")});
            background-size: cover;
            color: yellowgreen;; /* Cream color for text */
        }}
        p {{
            color: golden; /* Cream for body text */}}
        h1 {{
            color: #2E7D32; /* Dark green for agriculture theme */
        }}
        h2, h3 {{
            color: #4A2F1A; /* Earthy brown for subheaders */
        }}
        .stTextInput input {{
            background-color: #E8F5E9; /* Light green for input fields */
            color: #4A2F1A; /* Earthy brown for input text */
        }}
        .stMarkdown, .stText, .stCaption, .stTextInput > label, .stFileUploader > label, .stButton > button {{
            color: #FFF8E1; /* Cream for body text and labels for readability */
        }}
        .stSidebar .stMarkdown, .stSidebar .stText, .stSidebar .stSelectbox > label, .stSidebar .stCheckbox > label {{
            color: #FFF8E1; /* Cream for sidebar text */
        }}
        .stButton > button {{
            background-color: #689F38; /* Medium green for buttons */
            color: #FFF8E1; /* Cream text on buttons */
        }}
        .stButton > button:hover {{
            background-color: #8BC34A; /* Lighter green on hover */
        }}
        .stSpinner > div > div {{
            color: #FFF8E1; /* Cream for spinner text */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

run_app()

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: white; /* Light gray background for the app */
    }
   .sidebar .sidebar-content {
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_yolo_model():
    return YOLO(LEAF_COUNTER_MODEL)

try:
    st.set_page_config(page_title="Banana Plant Care Advisor", layout="wide")
    st.sidebar.selectbox("🔊 Select Language", options=list(LANGUAGES.keys()), key="selected_language")
except Exception as e:
    logger.error(f"Error setting up Streamlit config: {e}")

if not create_directories():
    st.error("Application setup failed: Could not create necessary directories.")
    st.stop()

model, embeddings, db = load_models()
qa_chain = load_llm(db)

try:
    st.title("🍌 Banana Plant Care Advisor")
    st.markdown('<h3 class="custom-subheader">Upload one crop image and one leaf image of your banana plant to get expert multilingual care advice including disease detection.</h3>', unsafe_allow_html=True)

    session_id = get_session_id()
    
    try:
        lang_code = LANGUAGES[st.session_state.get("selected_language", "English")]
    except Exception as e:
        logger.error(f"Error getting language code: {e}")
        lang_code = "en"

    # Debug mode toggle in sidebar
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)  # Default to True for troubleshooting

    with st.expander("Upload Plant Photos & Get Advice", expanded=True):
        st.header("Plant Analysis")
        
        try:
            name = st.text_input("Plant Name (Optional)", key="plant_name_input")
            age = st.text_input("Plant Age (e.g., '3 months')", key="plant_age_input")
            crop_file = st.file_uploader(
                "Upload Crop Image (whole plant or fruit)", 
                type=["png", "jpg", "jpeg", "webp"], 
                accept_multiple_files=False,
                key="crop_image"
            )
            leaf_file = st.file_uploader(
                "Upload Leaf Image", 
                type=["png", "jpg", "jpeg", "webp"], 
                accept_multiple_files=False,
                key="leaf_image"
            )
        except Exception as e:
            logger.error(f"Error creating input widgets: {e}")
            st.error("Error setting up input fields")

        if st.button("Analyze Plant"):
            try:
                if not crop_file or not leaf_file:
                    st.warning("Please upload both a crop image and a leaf image.")
                elif qa_chain is None:
                    st.error("AI service is not available. Please check configuration.")
                else:
                    with st.spinner("Analyzing your plant..."):
                        # Clear previous session state
                        if 'current_analysis' in st.session_state:
                            del st.session_state.current_analysis
                        analysis_id = str(uuid.uuid4())
                        result = {}

                        # Process crop image
                        try:
                            crop_bytes = crop_file.read()
                            crop_filename = crop_file.name
                            crop_unique_filename = f"{analysis_id}_crop_{crop_filename}"
                            crop_filepath = os.path.join(UPLOAD_FOLDER, crop_unique_filename)
                            
                            # Save crop file
                            with open(crop_filepath, "wb") as f:
                                f.write(crop_bytes)
                            logger.info(f"Crop file saved: {crop_filepath}")

                            # Validate crop image
                            img_check = cv2.imread(crop_filepath)
                            if img_check is None or img_check.shape[0] < 100 or img_check.shape[1] < 100:
                                logger.error(f"Invalid or low-resolution crop image: {crop_filepath}")
                                st.error("Invalid or low-resolution crop image. Please upload a valid image.")
                                raise ValueError("Invalid crop image")

                            # Detect features
                            banana_present = detect_banana(crop_bytes, model)
                            flower_present = detect_flower(crop_bytes)
                            stage = estimate_stage(banana_present, flower_present)

                            result.update({
                                "crop_image_path": crop_filepath,
                                "banana_detected": banana_present,
                                "flower_detected": flower_present,
                                "stage": stage,
                                "crop_filename": crop_filename
                            })
                            if debug_mode:
                                st.write(f"Debug: Crop analysis - Banana: {banana_present}, Flower: {flower_present}, Stage: {stage}")
                        except Exception as e:
                            logger.error(f"Error processing crop file: {e}")
                            st.error(f"Error processing crop image: {e}")
                            raise e

                        # Process leaf image
                        try:
                            leaf_bytes = leaf_file.read()
                            leaf_filename = leaf_file.name
                            leaf_unique_filename = f"{analysis_id}_leaf_{leaf_filename}"
                            leaf_filepath = os.path.join(UPLOAD_FOLDER, leaf_unique_filename)
                            
                            # Save leaf file
                            with open(leaf_filepath, "wb") as f:
                                f.write(leaf_bytes)
                            logger.info(f"Leaf file saved: {leaf_filepath}")

                            # Validate leaf image
                            img_check = cv2.imread(leaf_filepath)
                            if img_check is None or img_check.shape[0] < 100 or img_check.shape[1] < 100:
                                logger.error(f"Invalid or low-resolution leaf image: {leaf_filepath}, shape: {img_check.shape if img_check is not None else 'None'}")
                                st.error("Invalid or low-resolution leaf image. Please upload a valid image.")
                                raise ValueError("Invalid leaf image")

                            # Log image details
                            logger.info(f"Leaf image details: {leaf_filepath}, size: {len(leaf_bytes)} bytes, resolution: {img_check.shape}")

                            # Analyze leaf colors using provided function
                            try:
                                num_leaves, leaf_colors = analyze_leaf_colors(leaf_filepath)
                                logger.info(f"Leaf analysis result: {num_leaves} leaves, colors: {leaf_colors}")
                                leaf_summary = f"Detected {num_leaves} leaves with colors: {', '.join(leaf_colors)}"
                                if debug_mode:
                                    st.write(f"Debug: Leaf analysis - Leaves: {num_leaves}, Colors: {leaf_colors}")
                            except Exception as e:
                                logger.error(f"Error in leaf color analysis: {e}")
                                num_leaves = 0
                                leaf_colors = []
                                leaf_summary = "Leaf analysis failed"

                            # Disease detection for leaf only
                            try:
                                leaf_disease = predict_banana_disease(leaf_filepath)
                                leaf_disease_summary = f"Leaf disease: {leaf_disease}"
                                if debug_mode:
                                    st.write(f"Debug: Leaf disease - {leaf_disease}")
                            except Exception as e:
                                logger.error(f"Error in leaf disease detection: {e}")
                                leaf_disease = "Unknown"
                                leaf_disease_summary = "Leaf disease detection failed"

                            result.update({
                                "leaf_image_path": leaf_filepath,
                                "num_leaves": num_leaves,
                                "leaf_colors": leaf_colors,
                                "leaf_disease": leaf_disease,
                                "leaf_filename": leaf_filename
                            })
                        except Exception as e:
                            logger.error(f"Error processing leaf file: {e}")
                            st.error(f"Error processing leaf image: {e}")
                            raise e

                        # Generate query for RAG
                        query = f"""
                        Analyzing banana plant named '{name}' which is {age} old.
                        Fruits detected: {'Yes' if result.get('banana_detected', False) else 'No'}
                        Flowers detected: {'Yes' if result.get('flower_detected', False) else 'No'}
                        Estimated Stage: {result.get('stage', 'Unknown')}
                        Leaf Analysis: Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}
                        Leaf Disease: {result.get('leaf_disease', 'Unknown')}
                        Given this information, what care advice should be provided?
                        """

                        try:
                            response = qa_chain.invoke({"query": query})
                            advice = response.get('result', "No advice available.") if isinstance(response, dict) else str(response)
                            advice = comprehensive_text_cleaner(advice)
                            translated_advice = translate_text(advice, lang_code)
                            logger.info("AI response generated successfully")
                            result.update({
                                "advice": advice,
                                "translated_advice": translated_advice
                            })
                        except Exception as e:
                            logger.error(f"LLM/RAG chain error: {e}")
                            advice = "AI service temporarily unavailable. Please try again later."
                            translated_advice = translate_text(advice, lang_code)
                            result.update({
                                "advice": advice,
                                "translated_advice": translated_advice
                            })

                        # Save analysis
                        analysis_data = {
                            "analysis_id": analysis_id,
                            "name": name,
                            "age": age,
                            "result": result,
                            "questions": []
                        }

                        if save_to_history(session_id, analysis_data):
                            st.session_state.current_analysis = analysis_data
                            st.rerun()
                        else:
                            st.warning("Analysis completed but couldn't save to history")

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

            result = current_analysis.get('result', {})
            try:
                st.write("---")
                
                # Display crop image
                try:
                    if os.path.exists(result.get('crop_image_path', '')):
                        st.image(result['crop_image_path'], caption="Crop Image", width=300)
                    else:
                        st.warning("Crop image file not found")
                except Exception as e:
                    logger.error(f"Error displaying crop image: {e}")
                    st.warning("Could not display crop image")

                # Display leaf image
                try:
                    if os.path.exists(result.get('leaf_image_path', '')):
                        st.image(result['leaf_image_path'], caption="Leaf Image", width=300)
                    else:
                        st.warning("Leaf image file not found")
                except Exception as e:
                    logger.error(f"Error displaying leaf image: {e}")
                    st.warning("Could not display leaf image")

                st.write(f"**Banana Detected:** {'Yes' if result.get('banana_detected', False) else 'No'}")
                st.write(f"**Flower Detected:** {'Yes' if result.get('flower_detected', False) else 'No'}")
                st.write(f"**Estimated Stage:** {result.get('stage', 'Unknown')}")
                st.write(f"**Leaf Analysis:** Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}")
                st.write(f"**Leaf Disease:** {result.get('leaf_disease', 'Unknown')}")
                
                st.subheader("Care Advice:")
                advice_text = result.get('translated_advice', 'No advice available')
                st.write(advice_text)

                # Audio generation
                advice_audio_filepath = os.path.join(AUDIO_FOLDER, f"advice_{current_analysis['analysis_id']}.mp3")
                if st.button("Play Advice", key="play_advice"):
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
                logger.error(f"Error displaying result: {e}")
                st.error("Error displaying result")

        except Exception as e:
            logger.error(f"Error displaying current analysis: {e}")
            st.error("Error displaying analysis results")

    # Follow-up questions
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
                                question_en = question if lang_code == 'en' else translate_text(question, 'en')
                                response = qa_chain.invoke({"query": question_en})
                                answer = response.get('result', "No answer available.") if isinstance(response, dict) else str(response)
                                answer = comprehensive_text_cleaner(answer)
                                translated_answer = translate_text(answer, lang_code)

                                question_data = {
                                    "question": question,
                                    "answer": answer,
                                    "translated_answer": translated_answer,
                                    "timestamp": datetime.now().isoformat()
                                }
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

            try:
                if st.session_state.current_analysis.get('questions'):
                    st.subheader("Question History")
                    for i, q in enumerate(st.session_state.current_analysis['questions']):
                        try:
                            st.write(f"**Q{i+1}:** {q.get('question', 'Question not available')}")
                            answer_text = q.get('translated_answer', 'Answer not available')
                            st.write(f"**A{i+1}:** {answer_text}")
                            
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
        logger.error(f"Error in follow-up questions section: {e}")
        st.error("Error setting up follow-up questions")

except Exception as e:
    logger.error(f"Critical application error: {e}")
    logger.error(traceback.format_exc())
    st.error("Application encountered a critical error. Please refresh the page.")

try:
    st.markdown("---")
    st.markdown("### 🌱 Tips for Better Results")
    st.markdown("""
    - Upload clear, well-lit images of your banana plant
    - Ensure the crop image shows the whole plant or fruit clearly
    - Ensure the leaf image shows leaves clearly for accurate disease detection and color analysis
    - Provide plant age and name for more accurate advice
    - Ask specific questions about care concerns or detected leaf diseases
    """)
except Exception as e:
    logger.error(f"Error displaying footer: {e}")

try:
    if st.sidebar.button("🔧 System Status"):
        st.sidebar.write("**System Status:**")
        st.sidebar.write(f"✅ YOLO Model: {'Loaded' if model else 'Failed'}")
        st.sidebar.write(f"✅ Embeddings: {'Loaded' if embeddings else 'Failed'}")
        st.sidebar.write(f"✅ Database: {'Loaded' if db else 'Failed'}")
        st.sidebar.write(f"✅ LLM: {'Connected' if qa_chain else 'Failed'}")
        st.sidebar.write(f"✅ API Key: {'Configured' if os.getenv('GOOGLE_API_KEY') else 'Missing'}")
except Exception as e:
    logger.error(f"Error in system status: {e}")