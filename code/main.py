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
from config import LEAF_COUNTER_MODEL , BANANA_DISEASE_MODEL , BANANA_MODEL , BANANA_STAGE_MODEL
from datetime import datetime
from leaf_counter import analyze_leaf_colors


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
    st.markdown("Upload banana plant images and get expert multilingual care advice.")

    session_id = get_session_id()
    
    try:
        lang_code = LANGUAGES[st.session_state.get("selected_language", "English")]
    except Exception as e:
        logger.error(f"Error getting language code: {e}")
        lang_code = "en"

    
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
                                    banana_present = detect_banana(file_bytes, model)
                                    flower_present = detect_flower(file_bytes)
                                    stage = estimate_stage(banana_present, flower_present)
                                    
                                    # Analyze leaf colors
                                    try:
                                        num_leaves, leaf_colors = analyze_leaf_colors(filepath)
                                        leaf_summary = f"Detected {num_leaves} leaves with colors: {', '.join(leaf_colors)}"
                                    except Exception as e:
                                        logger.error(f"Error in leaf color analysis for {filename}: {e}")
                                        num_leaves = 0
                                        leaf_colors = []
                                        leaf_summary = "Leaf analysis failed"

                                    # Generate query for RAG
                                    query = f"""
                                    Analyzing banana plant named '{name}' which is {age} old.
                                    Fruits detected: {'Yes' if banana_present else 'No'}
                                    Flowers detected: {'Yes' if flower_present else 'No'}
                                    Estimated Stage: {stage}
                                    Leaf Analysis: {leaf_summary}
                                    Given this stage and leaf analysis, what care advice should be provided?
                                    """

                                    
                                    try:
                                        response = qa_chain.invoke({"query": query})
                                        advice = response.get('result', "No advice available.") if isinstance(response, dict) else str(response)
                                        advice = comprehensive_text_cleaner(advice)
                                        translated_advice = translate_text(advice, lang_code)
                                        logger.info("AI response generated successfully")
                                    except Exception as e:
                                        logger.error(f"LLM/RAG chain error: {e}")
                                        advice = "AI service temporarily unavailable. Please try again later."
                                        translated_advice = translate_text(advice, lang_code)

                                    results.append({
                                        "image_path": filepath,
                                        "banana_detected": banana_present,
                                        "flower_detected": flower_present,
                                        "stage": stage,
                                        "num_leaves": num_leaves,
                                        "leaf_colors": leaf_colors,
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
                                st.image(result['image_path'], caption=f"Image {i+1}", use_container_width=True,width=50)
                            else:
                                st.warning(f"Image {i+1} file not found")
                        except Exception as e:
                            logger.error(f"Error displaying image {i+1}: {e}")
                            st.warning(f"Could not display image {i+1}")
                        
                        st.write(f"**Banana Detected:** {'Yes' if result.get('banana_detected', False) else 'No'}")
                        st.write(f"**Flower Detected:** {'Yes' if result.get('flower_detected', False) else 'No'}")
                        st.write(f"**Estimated Stage:** {result.get('stage', 'Unknown')}")
                        st.write(f"**Leaf Analysis:** Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}")
                        
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
            logger.error(f"Error in follow-up questions tab: {e}")
            st.error("Error setting up follow-up questions")

except Exception as e:
    logger.error(f"Critical application error: {e}")
    logger.error(traceback.format_exc())
    st.error("Application encountered a critical error. Please refresh the page.")


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


try:
    if st.sidebar.button("🔧 System Status"):
        st.sidebar.write("**System Status:**")
        st.sidebar.write(f"✅ YOLO Model: {'Loaded' if model else '❌ Failed'}")
        st.sidebar.write(f"✅ Embeddings: {'Loaded' if embeddings else '❌ Failed'}")
        st.sidebar.write(f"✅ Database: {'Loaded' if db else '❌ Failed'}")
        st.sidebar.write(f"✅ LLM: {'Connected' if qa_chain else '❌ Failed'}")
        st.sidebar.write(f"✅ API Key: {'Configured' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
except Exception as e:
    logger.error(f"Error in system status: {e}")