import streamlit as st
import os
import sys
import traceback
import uuid
import requests
from datetime import datetime, timedelta
from config import logger, create_directories, LANGUAGES, UPLOAD_FOLDER, AUDIO_FOLDER
from models import load_models, load_llm 
from utils import (
    get_session_id, save_to_history, load_history, 
    translate_text, text_to_audio, comprehensive_text_cleaner
)
from detection import detect_banana, detect_flower, estimate_stage , detect_seedling
from config import LEAF_COUNTER_MODEL, BANANA_DISEASE_MODEL, BANANA_MODEL, BANANA_STAGE_MODEL , SEEDLING_MODEL
from leaf_counter import analyze_leaf_colors
from banana_disease_detection import predict_banana_disease
import cv2
from ultralytics import YOLO
import base64
import requests
from datetime import datetime, timedelta
import streamlit as st
import requests
from datetime import datetime, timedelta

st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(page_title="Home",layout="wide",initial_sidebar_state="auto",menu_items=None)  

hide_streamlit_style = """
            <style>
                /* Hide the Streamlit header and menu */
                header {visibility: hidden;}
                /* Optionally, hide the footer */
                .streamlit-footer {display: none;}
                /* Hide your specific div class, replace class name with the one you identified */
                .st-emotion-cache-uf99v8 {display: none;}
            </style>
            """
            
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/{image_file.split('.')[-1]};base64,{encoded_string}"


def get_weather_forecast(location):
    url = "https://weather-api167.p.rapidapi.com/api/weather/forecast"
    querystring = {f"place": {location}, "units": "metric"}
    headers = {
        "x-rapidapi-key": "46d33ff5a0mshe40b3178c84a8b4p1f5cf6jsnb84963ce1cd1",
        "x-rapidapi-host": "weather-api167.p.rapidapi.com",
        "Accept": "application/json"
    }

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    def calculate_gdd(temp_min, temp_max, base_temp=10):
        avg_temp = (temp_min + temp_max) / 2
        gdd = max(0, avg_temp - base_temp)
        return round(gdd, 2)

    def estimate_et(temp, humidity):
        et = max(0, (temp - 10) * (1 - humidity / 100) * 0.1)
        return round(et, 2)

    results = []
    if response.status_code == 200 and 'list' in data:
        for entry in data['list']:
            try:
                forecast_date = datetime.strptime(entry['dt_txt'], '%Y-%m-%d %H:%M:%S').date()
            except (ValueError, KeyError):
                return [{"error": "Invalid date format or missing dt_txt in response"}]

            if forecast_date in [yesterday, today, tomorrow]:
                main = entry['main']
                wind = entry['wind']
                clouds = entry['clouds']
                rain = entry.get('rain', {})
                weather_desc = entry['weather'][0]['description']

                temp = main['temprature']
                temp_min = main['temprature_min']
                temp_max = main['temprature_max']
                feels_like = main['temprature_feels_like']
                humidity = main['humidity']
                precipitation = rain.get('amount', 0)

                gdd = calculate_gdd(temp_min, temp_max)
                et = estimate_et(temp, humidity)
                frost_warning = "Yes" if temp_min <= 0 else "No"
                severe_weather = "Yes" if wind['speed'] > 10 or precipitation > 10 else "No"

                weather_data = {
                    "date_time": entry['dt_txt'],
                    "temperature": round(temp, 2),
                    "feels_like": round(feels_like, 2),
                    "temp_min": round(temp_min, 2),
                    "temp_max": round(temp_max, 2),
                    "humidity": humidity,
                    "precipitation": precipitation,
                    "wind_speed": wind['speed'],
                    "wind_direction": f"{wind['direction']} ({wind['degrees']}°)",
                    "cloud_cover": clouds['cloudiness'],
                    "frost_warning": frost_warning,
                    "gdd": gdd,
                    "evapotranspiration": et,
                    "severe_weather": severe_weather,
                    "description": weather_desc
                }
                results.append(weather_data)
        return results
    else:
        return [{"error": f"Error fetching data: {data.get('message', 'Unknown error')}"}]

# Run and print result
def format_weather_for_ai(weather_data, location):
    """Format weather data for AI consumption"""
    if not weather_data or weather_data[0].get('error'):
        return f"Weather data for {location} is currently unavailable."
    
    weather_summary = f"Weather forecast for {location}:\n"
    for day_data in weather_data:
        weather_summary += f"""
Date: {day_data['date_time']}
Temperature: {day_data['temperature']}°C (Min: {day_data['temp_min']}°C, Max: {day_data['temp_max']}°C)
Humidity: {day_data['humidity']}%
Precipitation: {day_data['precipitation']}mm
Wind: {day_data['wind_speed']} m/s
Growing Degree Days: {day_data['gdd']}
Evapotranspiration: {day_data['evapotranspiration']}mm
Frost Warning: {day_data['frost_warning']}
Severe Weather: {day_data['severe_weather']}
Description: {day_data['description']}
---
"""
    return weather_summary



def build_conversation_context(current_analysis):
    """Build conversation context including previous questions and answers"""
    context = ""
    
    # Add plant analysis context
    if current_analysis:
        result = current_analysis.get('result', {})
        context += f"""
Plant Analysis Context:
- Plant Name: {current_analysis.get('name', 'Unknown')}
- Plant Age: {current_analysis.get('age', 'Unknown')}
- Banana Detected: {'Yes' if result.get('banana_detected', False) else 'No'}
- Flower Detected: {'Yes' if result.get('flower_detected', False) else 'No'}
- Estimated Stage: {result.get('stage', 'Unknown')}
- Leaf Analysis: Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}
- Leaf Disease: {result.get('leaf_disease', 'Unknown')}

Previous Advice Given:
{result.get('advice', 'No previous advice')}

"""
        
        # Add previous questions and answers
        questions = current_analysis.get('questions', [])
        if questions:
            context += "Previous Questions and Answers:\n"
            for i, qa in enumerate(questions, 1):
                context += f"Q{i}: {qa.get('question', 'No question')}\n"
                context += f"A{i}: {qa.get('answer', 'No answer')}\n\n"
    
    return context



def run_app():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url({add_bg_from_local("static/styles/bg/background_image.jpg")});
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
        .st-emotion-cache-zy6yx3 {{
            padding: 0rem 5rem 1rem;;
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
    # st.selectbox("🔊 Select Language", options=list(LANGUAGES.keys()), key="selected_language")
except Exception as e:
    logger.error(f"Error setting up Streamlit config: {e}")

if not create_directories():
    st.error("Application setup failed: Could not create necessary directories.")
    st.stop()

model, embeddings, db = load_models()
qa_chain = load_llm(db)

try:
    col1 , col2 , col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🍌 Banana Plant Care Advisor")
    with col3:
        st.selectbox("🔊 Select Language", options=list(LANGUAGES.keys()), key="selected_language")
    st.markdown('<h3 class="custom-subheader">Upload crop and leaf images of your banana plant to get expert multilingual care advice with weather-based recommendations.</h3>', unsafe_allow_html=True)

    session_id = get_session_id()
    
    try:
        lang_code = LANGUAGES[st.session_state.get("selected_language", "English")]
    except Exception as e:
        logger.error(f"Error getting language code: {e}")
        lang_code = "en"

    # Debug mode toggle in sidebar
    # debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

    with st.expander("Upload Plant Photos & Get Advice", expanded=True):
        st.header("Plant Analysis")
        
        try:
            name = st.text_input("Plant Name (Optional)", key="plant_name_input")
            age_options = [f"{i} month{'s' if i > 1 else ''}" for i in range(1, 13)] + ["1 year", "2 years", "Custom"]
            selected_age = st.selectbox("Plant Age", options=age_options, key="plant_age_select")


            if selected_age == "Custom":
                custom_age = st.text_input("Enter Custom Plant Age (e.g., '3 months', '5 years')", key="plant_age_input")
                age = custom_age
            else:
                age = selected_age
            location = st.text_input("Location (for weather data)", placeholder="e.g., Mumbai, India", key="location_input")
            
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

        # Inside the "Analyze Plant" button block
        if st.button("Analyze Plant"):
            try:
                if not crop_file or not leaf_file:
                    st.warning("Please upload both a crop image and a leaf image.")
                elif qa_chain is None:
                    st.error("AI service is not available. Please check configuration.")
                else:
                    with st.spinner("Analyzing your plant and fetching weather data..."):
                        # Clear previous session state
                        if 'current_analysis' in st.session_state:
                            del st.session_state.current_analysis
                        analysis_id = str(uuid.uuid4())
                        result = {}

                        # Get weather data if location provided
                        weather_context = ""
                        if location:
                            try:
                                weather_data = get_weather_forecast(location)
                                weather_context = format_weather_for_ai(weather_data, location.strip())
                                result['weather_data'] = weather_data
                                logger.info(f"Weather data fetched for {location}: {weather_data[:1]}")
                            except Exception as e:
                                logger.error(f"Weather fetch error: {e}")
                                weather_context = f"Weather data unavailable for {location}"

                        # Process crop image
                        try:
                            crop_bytes = crop_file.read()
                            crop_filename = crop_file.name
                            crop_unique_filename = f"{analysis_id}_crop_{crop_filename}"
                            crop_filepath = os.path.join(UPLOAD_FOLDER, crop_unique_filename)

                            with open(crop_filepath, "wb") as f:
                                f.write(crop_bytes)
                            logger.info(f"Crop file saved: {crop_filepath}")

                            img_check = cv2.imread(crop_filepath)
                            if img_check is None or img_check.shape[0] < 100 or img_check.shape[1] < 100:
                                logger.error(f"Invalid or low-resolution crop image: {crop_filepath}")
                                st.error("Invalid or low-resolution crop image. Please upload a valid image.")
                                raise ValueError("Invalid crop image")

                            banana_present = detect_banana(crop_bytes, model)
                            flower_present = detect_flower(crop_bytes)
                            seedling_present = detect_seedling(crop_bytes, SEEDLING_MODEL)
                            # Pass age to estimate_stage
                            stage = estimate_stage(banana_present, flower_present, seedling_present, age=age)

                            result.update({
                                "crop_image_path": crop_filepath,
                                "banana_detected": banana_present,
                                "flower_detected": flower_present,
                                "stage": stage,
                                "crop_filename": crop_filename
                            })

                        except Exception as e:
                            logger.error(f"Error processing crop file: {e}")
                            st.error(f"Error processing crop image: {e}")
                            raise e

                        try:
                            leaf_bytes = leaf_file.read()
                            leaf_filename = leaf_file.name
                            leaf_unique_filename = f"{analysis_id}_leaf_{leaf_filename}"
                            leaf_filepath = os.path.join(UPLOAD_FOLDER, leaf_unique_filename)
                            
                            with open(leaf_filepath, "wb") as f:
                                f.write(leaf_bytes)
                            logger.info(f"Leaf file saved: {leaf_filepath}")

                            img_check = cv2.imread(leaf_filepath)
                            if img_check is None or img_check.shape[0] < 100 or img_check.shape[1] < 100:
                                logger.error(f"Invalid or low-resolution leaf image: {leaf_filepath}")
                                st.error("Invalid or low-resolution leaf image. Please upload a valid image.")
                                raise ValueError("Invalid leaf image")

                            try:
                                num_leaves, leaf_colors = analyze_leaf_colors(leaf_filepath)
                                logger.info(f"Leaf analysis result: {num_leaves} leaves, colors: {leaf_colors}")

                            except Exception as e:
                                logger.error(f"Error in leaf color analysis: {e}")
                                num_leaves = 0
                                leaf_colors = []

                            try:
                                leaf_disease = predict_banana_disease(leaf_filepath)

                            except Exception as e:
                                logger.error(f"Error in leaf disease detection: {e}")
                                leaf_disease = "Unknown"

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

                        # Enhanced query for RAG with weather integration
                        query = f"""
As an expert agricultural advisor, analyze the provided banana plant data and deliver comprehensive, stage-specific care advice based on agricultural best practices, current weather conditions, and the 'Banana Plant Life Cycle Guide' from Bandhan Agritech Private Limited[](https://bandhanagri.com). The growth stage is primarily determined by the plant's age ({age}), refined by image analysis. All recommendations must align with the guide’s fertilizer, pest management, and sustainability practices, referencing specific products (e.g., BioStart, Bamida) where applicable.

PLANT ANALYSIS:
- Plant Name: '{name}'
- Plant Age: {age}
- Fruits Detected: {'Yes' if result.get('banana_detected', False) else 'No'}
- Flowers Detected: {'Yes' if result.get('flower_detected', False) else 'No'}
- Estimated Growth Stage: {result.get('stage', 'Unknown')}
- Leaf Analysis: Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}
- Leaf Disease Status: {result.get('leaf_disease', 'Unknown')}

WEATHER CONDITIONS:
{weather_context}

Provide detailed care advice addressing:
1. **Plant Health and Growth Stage**: Tailor recommendations to the age-based growth stage ({result.get('stage', 'Unknown')}), detailing:
   - Optimal environmental conditions (temperature, humidity, sunlight) from the guide.
   - Fertilizer applications (e.g., product, rate, method, timing) for the stage, including organic options.
   - Physiological needs (e.g., root development, fruit filling) and monitoring techniques.
2. **Weather Impact**: Adjust care based on weather conditions, including:
   - Watering schedule (liters/week or mm/week) considering evapotranspiration rates and rainfall.
   - Temperature and humidity management (e.g., misting, shade nets).
   - Wind protection measures (e.g., windbreaks) if high winds are reported.
3. **Disease and Pest Management**: Recommend prevention/treatment based on leaf disease status and stage-specific risks, using Bandhan Agritech insecticides (e.g., Bamida, WeevilGuard, FruitSafe, EcoShield):
   - Specify product, application rate (e.g., mL/ha), and timing.
   - Include integrated pest management (IPM) practices (e.g., biological controls, cultural methods).
4. **Frost Protection**: If frost risk exists (temperatures below , suggest protective measures (e.g., mulching, frost blankets).
5. **Immediate Actions**: Highlight urgent tasks (e.g., pest treatment, irrigation adjustments) based on weather alerts or plant health.
6. **AI-Driven Insights**: Provide:
   - Yield protection estimates (e.g., tonnes/ha gain from interventions).
   - Cost-benefit analysis for fertilizers/pesticides
   - Optimization tips (e.g., precision agriculture tools, soil testing frequency).

Ensure recommendations are:
- **Stage-Specific**: Align with the guide’s stages (Germination, Seedling, Vegetative, Flowering, Fruit Development, Harvesting).
- **Sustainable**: Prioritize organic fertilizers (e.g., vermicompost) and IPM, referencing the guide’s sustainability practices.
- **Weather-Adapted**: Account for current temperature, humidity, rainfall, and alerts in {weather_context}.

Base advice on the 'Banana Plant Life Cycle Guide,' Bandhan Agritech’s product catalog, and real-time weather data, ensuring practical, actionable recommendations for farmers.
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
                            "location": location.strip() if location.strip() else "Not provided",
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
            st.write(f"**Location:** {current_analysis.get('location', 'Not provided')}")

            result = current_analysis.get('result', {})
            
            # Display weather information if available
            if 'weather_data' in result and result['weather_data']:
                weather_data = result['weather_data']
                if not weather_data[0].get('error'):
                    st.subheader("🌤️ Weather Forecast")
                    for day_data in weather_data[:3]:  # Show max 3 days
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Temperature", f"{day_data['temperature']}°C")
                        with col2:
                            st.metric("Humidity", f"{day_data['humidity']}%")
                        with col3:
                            st.metric("GDD", f"{day_data['gdd']}")
                        with col4:
                            st.metric("Precipitation", f"{day_data['precipitation']}mm")
                        
                        if day_data['frost_warning'] == 'Yes':
                            st.warning("⚠️ Frost Warning - Protect your plants!")
                        if day_data['severe_weather'] == 'Yes':
                            st.warning("⚠️ Severe Weather Alert")
                        st.write(f"**Conditions:** {day_data['description']}")
                        st.write("---")

            try:
                # Display crop image
                if os.path.exists(result.get('crop_image_path', '')):
                    st.image(result['crop_image_path'], caption="Crop Image", width=300)
                
                # Display leaf image
                if os.path.exists(result.get('leaf_image_path', '')):
                    st.image(result['leaf_image_path'], caption="Leaf Image", width=300)

                st.write(f"**Banana Detected:** {'Yes' if result.get('banana_detected', False) else 'No'}")
                st.write(f"**Flower Detected:** {'Yes' if result.get('flower_detected', False) else 'No'}")
                st.write(f"**Estimated Stage:** {result.get('stage', 'Unknown')}")
                st.write(f"**Leaf Analysis:** Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}")
                
                st.write(f"**Leaf Disease:** {result.get('leaf_disease', 'Unknown')}")
                # st.write(f"**Weather Report:** {weather_data}")
                
                st.subheader("🌱 Care Advice (Weather-Enhanced):")
                advice_text = result.get('translated_advice', 'No advice available')
                st.markdown(advice_text)

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

    # Enhanced Follow-up questions with conversation memory
    st.header("Follow-up Questions")
    try:
        if 'current_analysis' in st.session_state:
            question = st.text_area("Ask a follow-up question about your plant (I remember our conversation):", key="follow_up_question")
            if st.button("Submit Question"):
                try:
                    if not question.strip():
                        st.warning("Please enter a question.")
                    elif qa_chain is None:
                        st.error("AI service is not available.")
                    else:
                        with st.spinner("Processing your question..."):
                            try:
                                # Build conversation context
                                conversation_context = build_conversation_context(st.session_state.current_analysis)
                                
                                # Get weather context if available
                                current_analysis = st.session_state.current_analysis
                                weather_context = ""
                                if 'result' in current_analysis and 'weather_data' in current_analysis['result']:
                                    location = current_analysis.get('location', '')
                                    weather_context = format_weather_for_ai(current_analysis['result']['weather_data'], location)
                                
                                # Enhanced query with full context
                                question_en = question if lang_code == 'en' else translate_text(question, 'en')
                                enhanced_query = f"""
                                {conversation_context}
                                
                                Current Weather Context:
                                {weather_context}
                                
                                NEW QUESTION: {question_en}
                                
                                Please answer this new question considering:
                                1. All the previous plant analysis and advice given
                                2. Previous questions and answers in our conversation
                                3. Current weather conditions and their impact
                                4. Provide specific, actionable advice based on the complete context
                                
                                Remember to reference previous advice when relevant and build upon our conversation history.
                                """
                                
                                response = qa_chain.invoke({"query": enhanced_query})
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
    - Ensure the leaf image shows leaves clearly for accurate disease detection
    - Provide plant age, name, and location for personalized weather-based advice
    - Ask specific questions about care concerns - I remember our entire conversation!
    - Check weather alerts and frost warnings for immediate plant protection
    """)
except Exception as e:
    logger.error(f"Error displaying footer: {e}")

