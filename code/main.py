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
from detection import detect_banana, detect_flower, estimate_stage, detect_seedling
from config import LEAF_COUNTER_MODEL, BANANA_DISEASE_MODEL, BANANA_MODEL, BANANA_STAGE_MODEL, SEEDLING_MODEL
from leaf_counter import analyze_leaf_colors
from banana_disease_detection import predict_banana_disease, predict_banana_disease_yolo
import cv2
from ultralytics import YOLO
import base64

# Clear caches
st.cache_data.clear()
st.cache_resource.clear()

# Define English UI strings
UI_STRINGS = {
    "page_title": "Banana Plant Care Advisor",
    "app_title": "🍌 Banana Plant Care Advisor",
    "subheader": "Upload crop and leaf images of your banana plant to get expert multilingual care advice with weather-based recommendations.",
    "plant_analysis": "Plant Analysis",
    "plant_name_label": "Plant Name (Optional)",
    "plant_age_label": "Plant Age",
    "location_label": "Location (for weather data)",
    "location_placeholder": "e.g., Mumbai, India",
    "crop_image_label": "Upload Crop Image (whole plant or fruit)",
    "leaf_image_label": "Upload Leaf Image",
    "analyze_button": "Analyze Plant",
    "play_advice_button": "Play Advice",
    "follow_up_header": "Follow-up Questions",
    "follow_up_placeholder": "Ask a follow-up question about your plant (I remember our conversation):",
    "submit_question_button": "Submit Question",
    "latest_results": "Latest Analysis Results",
    "plant_name": "Plant Name",
    "plant_age": "Plant Age",
    "location": "Location",
    "weather_forecast": "🌤️ Weather Forecast",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "gdd": "GDD",
    "precipitation": "Precipitation",
    "frost_warning": "⚠️ Frost Warning - Protect your plants!",
    "severe_weather": "⚠️ Severe Weather Alert",
    "conditions": "Conditions",
    "banana_detected": "Banana Detected",
    "flower_detected": "Flower Detected",
    "estimated_stage": "Estimated Stage",
    "leaf_analysis": "Leaf Analysis",
    "leaf_disease": "Leaf Disease",
    "care_advice": "🌱 Care Advice (Weather-Enhanced):",
    "tips_header": "🌱 Tips for Better Results",
    "tips_content": """
- Upload clear, well-lit images of your banana plant
- Ensure the crop image shows the whole plant or fruit clearly
- Ensure the leaf image shows leaves clearly for accurate disease detection
- Provide plant age, name, and location for personalized weather-based advice
- Ask specific questions about care concerns - I remember our entire conversation!
- Check weather alerts and frost warnings for immediate plant protection
    """,
    "no_images_warning": "Please upload both a crop image and a leaf image.",
    "ai_unavailable": "AI service is not available. Please check configuration.",
    "invalid_crop_image": "Invalid or low-resolution crop image. Please upload a valid image.",
    "invalid_leaf_image": "Invalid or low-resolution leaf image. Please upload a valid image.",
    "analysis_failed": "Analysis failed: {error}",
    "history_save_warning": "Analysis completed but couldn't save to history",
    "no_question_warning": "Please enter a question.",
    "question_error": "Error processing your question. Please try again.",
    "audio_failed": "Audio generation failed. Please try again.",
    "no_analysis_info": "Please analyze a plant first to ask follow-up questions.",
    "question_history": "Question History",
    "app_error": "Application encountered a critical error. Please refresh the page.",
    "language_label": "🔊 Select Language",
    "audio_caption": "🔈 Language: {lang}"
}

# Function to get translated UI strings (retained for potential future use)
def get_translated_ui_strings(lang_code):
    if lang_code == "en":
        return UI_STRINGS
    translated = {}
    for key, text in UI_STRINGS.items():
        translated[key] = translate_text(text, lang_code)
    return translated

# Set page config
st.set_page_config(page_title=UI_STRINGS["page_title"], layout="wide", initial_sidebar_state="auto", menu_items=None)

# Hide Streamlit header and footer
hide_streamlit_style = """
    <style>
        /* Hide the Streamlit header and menu */
        header {visibility: hidden;}
        /* Optionally, hide the footer */
        .streamlit-footer {display: none;}
        /* Hide specific div class */
        .st-emotion-cache-uf99v8 {display: none;}
        .st-emotion-cache-1r61a0z {background-color: transparent;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/{image_file.split('.')[-1]};base64,{encoded_string}"

def get_weather_forecast(location):
    url = "https://weather-api167.p.rapidapi.com/api/weather/forecast"
    querystring = {"place": location, "units": "metric"}
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

                temp = main['temperature']
                temp_min = main['temp_min']
                temp_max = main['temp_max']
                feels_like = main['temperature_feels_like']
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

def format_weather_for_ai(weather_data, location, lang_code):
    if not weather_data or weather_data[0].get('error'):
        return translate_text(f"Weather data for {location} is currently unavailable.", lang_code)
    
    weather_summary = translate_text(f"Weather forecast for {location}:\n", lang_code)
    for day_data in weather_data:
        weather_summary += translate_text(f"""
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
""", lang_code)
    return weather_summary

def build_conversation_context(current_analysis, lang_code):
    context = ""
    if current_analysis:
        result = current_analysis.get('result', {})
        context += translate_text(f"""
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

""", lang_code)
        
        questions = current_analysis.get('questions', [])
        if questions:
            context += translate_text("Previous Questions and Answers:\n", lang_code)
            for i, qa in enumerate(questions, 1):
                context += translate_text(f"Q{i}: {qa.get('question', 'No question')}\n", lang_code)
                context += translate_text(f"A{i}: {qa.get('answer', 'No answer')}\n\n", lang_code)
    
    return context

def run_app():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url({add_bg_from_local("static/styles/bg/background_image.jpg")});
            background-size: cover;
            color: yellowgreen;
        }}
        p {{
            color: golden;
        }}
        h1 {{
            color: #2E7D32;
        }}
        h2, h3 {{
            color: #4A2F1A;
        }}
        .stTextInput input {{
            background-color: #E8F5E9;
            color: #4A2F1A;
        }}
        .stMarkdown, .stText, .stCaption, .stTextInput > label, .stFileUploader > label, .stButton > button {{
            color: #FFF8E1;
        }}
        .stSidebar .stMarkdown, .stSidebar .stText, .stSidebar .stSelectbox > label, .stSidebar .stCheckbox > label {{
            color: #FFF8E1;
        }}
        .stButton > button {{
            background-color: #689F38;
            color: #FFF8E1;
        }}
        .stButton > button:hover {{
            background-color: #8BC34A;
        }}
        .stSpinner > div > div {{
            color: #FFF8E1;
        }}
        .st-emotion-cache-zy6yx3 {{
            padding: 0rem 5rem 1rem;
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
        background-color: white;
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
    # Get language code
    lang_code = LANGUAGES.get(st.session_state.get("selected_language", "English"), "en")

    # Set page config with English title
    st.set_page_config(page_title=UI_STRINGS["page_title"], layout="wide")

    if not create_directories():
        st.error(UI_STRINGS["app_error"])
        st.stop()

    model, embeddings, db = load_models()
    qa_chain = load_llm(db)

    try:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title(UI_STRINGS["app_title"])
        with col3:
            st.selectbox(UI_STRINGS["language_label"], options=list(LANGUAGES.keys()), key="selected_language")
        st.markdown(f'<h3 class="custom-subheader">{UI_STRINGS["subheader"]}</h3>', unsafe_allow_html=True)

        session_id = get_session_id()

        with st.expander(UI_STRINGS["plant_analysis"], expanded=True):
            st.header(UI_STRINGS["plant_analysis"])
            
            try:
                name = st.text_input(UI_STRINGS["plant_name_label"], key="plant_name_input")
                age_options = [f"{i} month{'s' if i > 1 else ''}" for i in range(1, 13)] + ["1 year", "2 years", "Custom"]
                selected_age = st.selectbox(UI_STRINGS["plant_age_label"], options=age_options, key="plant_age_select")

                if selected_age == "Custom":
                    custom_age = st.text_input("Enter Custom Plant Age (e.g., '3 months', '5 years')", key="plant_age_input")
                    age = custom_age
                else:
                    age = selected_age
                location = st.text_input(UI_STRINGS["location_label"], placeholder=UI_STRINGS["location_placeholder"], key="location_input")
                
                water_given = st.text_input("Last Time Water Given (in liters) with Time", placeholder="e.g., 5 liters at 8 am", key="water_given_input")
                crop_file = st.file_uploader(
                    UI_STRINGS["crop_image_label"], 
                    type=["png", "jpg", "jpeg", "webp"], 
                    accept_multiple_files=False,
                    key="crop_image"
                )
                leaf_file = st.file_uploader(
                    UI_STRINGS["leaf_image_label"], 
                    type=["png", "jpg", "jpeg", "webp"], 
                    accept_multiple_files=False,
                    key="leaf_image"
                )
            except Exception as e:
                logger.error(f"Error creating input widgets: {e}")
                st.error(UI_STRINGS["app_error"])

            if st.button(UI_STRINGS["analyze_button"]):
                st.cache_data.clear()
                st.cache_resource.clear()
                try:
                    if not crop_file or not leaf_file:
                        st.warning(UI_STRINGS["no_images_warning"])
                    elif qa_chain is None:
                        st.error(UI_STRINGS["ai_unavailable"])
                    else:
                        with st.spinner("Analyzing your plant and fetching weather data..."):
                            if 'current_analysis' in st.session_state:
                                del st.session_state.current_analysis
                            analysis_id = str(uuid.uuid4())
                            result = {}

                            weather_context = ""
                            if location:
                                try:
                                    weather_data = get_weather_forecast(location)
                                    weather_context = format_weather_for_ai(weather_data, location.strip(), lang_code)
                                    result['weather_data'] = weather_data
                                    logger.info(f"Weather data fetched for {location}: {weather_data[:1]}")
                                except Exception as e:
                                    logger.error(f"Weather fetch error: {e}")
                                    weather_context = translate_text(f"Weather data unavailable for {location}", lang_code)

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
                                    st.error(UI_STRINGS["invalid_crop_image"])
                                    raise ValueError("Invalid crop image")

                                banana_present = detect_banana(crop_bytes, model)
                                flower_present = detect_flower(crop_bytes)
                                seedling_present = detect_seedling(crop_bytes)
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
                                st.error(UI_STRINGS["analysis_failed"].format(error=str(e)))
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
                                    st.error(UI_STRINGS["invalid_leaf_image"])
                                    raise ValueError("Invalid leaf image")

                                try:
                                    num_leaves, leaf_colors = analyze_leaf_colors(leaf_filepath)
                                    logger.info(f"Leaf analysis result: {num_leaves} leaves, colors: {leaf_colors}")
                                except Exception as e:
                                    logger.error(f"Error in leaf color analysis: {e}")
                                    num_leaves = 0
                                    leaf_colors = []

                                try:
                                    annotated_leaf_path = os.path.splitext(leaf_filepath)[0] + "_annotated.jpg"
                                    leaf_disease = predict_banana_disease_yolo(leaf_filepath, annotated_leaf_path)
                                except Exception as e:
                                    logger.error(f"Error in leaf disease detection: {e}")
                                    leaf_disease = "Unknown"

                                result.update({
                                    "leaf_image_path": leaf_filepath,
                                    "num_leaves": num_leaves,
                                    "leaf_colors": leaf_colors,
                                    "leaf_disease": leaf_disease,
                                    "leaf_filename": leaf_filename,
                                    "leaf_image_path": annotated_leaf_path,
                                })
                            except Exception as e:
                                logger.error(f"Error processing leaf file: {e}")
                                st.error(UI_STRINGS["analysis_failed"].format(error=str(e)))
                                raise e

                            query = f"""
As an expert agricultural advisor, analyze the provided banana plant data and deliver comprehensive, stage-specific care advice based on agricultural best practices, current weather conditions, and the 'Banana Plant Life Cycle Guide' from Bandhan Agritech Private Limited[](https://bandhanagri.com). The growth stage is primarily determined by the plant's age ({age}), refined by image analysis. All recommendations must align with the guide’s fertilizer, pest management, and sustainability practices, referencing specific products (e.g., BioStart, Bamida) where applicable.

PLANT ANALYSIS:
- Plant Name: '{name}'
- Plant Age: {age}
- Last Time Water Given with Amount: {water_given} liters
- Fruits Detected: {'Yes' if result.get('banana_detected', False) else 'No'}
- Flowers Detected: {'Yes' if result.get('flower_detected', False) else 'No'}
- Estimated Growth Stage: {result.get('stage', 'Unknown')}
- Leaf Analysis: Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}
- Leaf Disease Status: {result.get('leaf_disease', 'Unknown')}

WEATHER CONDITIONS:
{weather_context}

**Disease Validation**:
- Verify if the detected Leaf Disease Status ({result.get('leaf_disease', 'Unknown')}) is plausible for the Estimated Growth Stage ({result.get('stage', 'Unknown')}) based on general agricultural knowledge and the 'Banana Plant Life Cycle Guide'. If the disease is unlikely for the stage, note potential misidentification or contributing factors (e.g., environmental stress, image quality) and adjust recommendations accordingly.

**Structured Output for Disease Cases**:
If the Leaf Disease Status is not 'Unknown' or 'Healthy', structure the response as follows:
1. **Information about the Disease**:
   - Provide a detailed description of the disease, including symptoms, causes (e.g., fungal, bacterial, viral), and typical conditions (e.g., humidity, temperature) that promote it.
2. **Prevention from Disease**:
   - Recommend preventive measures to avoid spread or recurrence, including cultural practices (e.g., spacing, pruning), integrated pest management (IPM) techniques, and environmental adjustments.
3. **Cure and Suggested Products**:
   - Suggest specific treatments using Bandhan Agritech products (e.g., Bamida, WeevilGuard, FruitSafe, EcoShield), specifying product name, application rate (e.g., mL/ha), method, and timing.
   - Include organic or sustainable treatment options where applicable.
4. **Rest of the Advice**:
   - Provide the remaining care advice as detailed below, formatted as normal text.

**General Care Advice** (for all cases, placed after disease section if applicable):
Provide detailed care advice addressing:
1. **Plant Health and Growth Stage**: Tailor recommendations to the age-based growth stage ({result.get('stage', 'Unknown')}), detailing:
   - Optimal environmental conditions (temperature, humidity, sunlight) from the guide.
   - Fertilizer applications (e.g., product, rate, method, timing) for the stage, including organic options.
   - Physiological needs (e.g., root development, fruit filling) and monitoring techniques.
2. **Weather Impact**: Adjust care based on weather conditions, including:
   - Watering schedule (liters/week or mm/week) considering evapotranspiration rates, rainfall, and the last watering event ({water_given} liters).
   - Specific recommendation for the next watering time (e.g., date or time interval) based on the plant’s growth stage, weather forecast, and prior watering.
   - Temperature and humidity management (e.g., misting, shade nets).
   - Wind protection measures (e.g., windbreaks) if high winds are reported.
3. **Disease and Pest Management** (if no specific disease is detected): Recommend prevention/treatment for stage-specific risks, using Bandhan Agritech insecticides (e.g., Bamida, WeevilGuard, FruitSafe, EcoShield):
   - Specify product, application rate (e.g., mL/ha), and timing.
   - Include integrated pest management (IPM) practices (e.g., biological controls, cultural methods).
4. **Frost Protection**: If frost risk exists (temperatures below 0°C), suggest protective measures (e.g., mulching, frost blankets).
5. **Immediate Actions**: Highlight urgent tasks (e.g., pest treatment, irrigation adjustments) based on weather alerts or plant health.
6. **AI-Driven Insights**: Provide:
   - Yield protection estimates (e.g., tonnes/ha gain from interventions).
   - Cost-benefit analysis for fertilizers/pesticides.
   - Optimization tips (e.g., precision agriculture tools, soil testing frequency).

**Output Requirements**:
- Ensure recommendations are:
  - **Stage-Specific**: Align with the guide’s stages (Seedling, Vegetative, Flowering, Fruit Development, Harvesting).
  - **Sustainable**: Prioritize organic fertilizers (e.g., vermicompost) and IPM, referencing the guide’s sustainability practices.
  - **Weather-Adapted**: Account for current temperature, humidity, rainfall, alerts, and the last watering event in {weather_context}.
- Base advice on the 'Banana Plant Life Cycle Guide,' Bandhan Agritech’s product catalog, and real-time weather data, ensuring practical, actionable recommendations for farmers.
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
                                advice = translate_text("AI service temporarily unavailable. Please try again later.", lang_code)
                                translated_advice = advice
                                result.update({
                                    "advice": advice,
                                    "translated_advice": translated_advice
                                })

                            analysis_data = {
                                "analysis_id": analysis_id,
                                "name": name,
                                "age": age,
                                "location": location.strip() if location.strip() else translate_text("Not provided", lang_code),
                                "result": result,
                                "questions": []
                            }

                            if save_to_history(session_id, analysis_data):
                                st.session_state.current_analysis = analysis_data
                                st.rerun()
                            else:
                                st.warning(UI_STRINGS["history_save_warning"])

                except Exception as e:
                    logger.error(f"Critical error in plant analysis: {e}")
                    logger.error(traceback.format_exc())
                    st.error(UI_STRINGS["analysis_failed"].format(error=str(e)))

        if 'current_analysis' in st.session_state:
            try:
                current_analysis = st.session_state.current_analysis
                st.subheader(UI_STRINGS["latest_results"])
                st.write(f"**{UI_STRINGS['plant_name']}:** {current_analysis.get('name', 'Unknown')}")
                st.write(f"**{UI_STRINGS['plant_age']}:** {current_analysis.get('age', 'Unknown')}")
                st.write(f"**{UI_STRINGS['location']}:** {current_analysis.get('location', 'Not provided')}")

                result = current_analysis.get('result', {})
                
                if 'weather_data' in result and result['weather_data']:
                    weather_data = result['weather_data']
                    if not weather_data[0].get('error'):
                        st.subheader(UI_STRINGS["weather_forecast"])
                        for day_data in weather_data[:3]:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(UI_STRINGS["temperature"], f"{day_data['temperature']}°C")
                            with col2:
                                st.metric(UI_STRINGS["humidity"], f"{day_data['humidity']}%")
                            with col3:
                                st.metric(UI_STRINGS["gdd"], f"{day_data['gdd']}")
                            with col4:
                                st.metric(UI_STRINGS["precipitation"], f"{day_data['precipitation']}mm")
                            
                            if day_data['frost_warning'] == 'Yes':
                                st.warning(UI_STRINGS["frost_warning"])
                            if day_data['severe_weather'] == 'Yes':
                                st.warning(UI_STRINGS["severe_weather"])
                            st.write(f"**{UI_STRINGS['conditions']}:** {translate_text(day_data['description'], lang_code)}")
                            st.write("---")

                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        if os.path.exists(result.get('crop_image_path', '')):
                            st.image(result['crop_image_path'], caption="Crop Image", width=300)
                    
                    with col2:
                        if os.path.exists(result.get('leaf_image_path', '')):
                            st.image(result['leaf_image_path'], caption="Leaf Image")

                    st.write(f"**{UI_STRINGS['banana_detected']}:** {translate_text('Yes' if result.get('banana_detected', False) else 'No', lang_code)}")
                    st.write(f"**{UI_STRINGS['flower_detected']}:** {translate_text('Yes' if result.get('flower_detected', False) else 'No', lang_code)}")
                    st.write(f"**{UI_STRINGS['estimated_stage']}:** {translate_text(result.get('stage', 'Unknown'), lang_code)}")
                    leaf_analysis_text = f"Detected {result.get('num_leaves', 0)} leaves with colors: {', '.join(result.get('leaf_colors', []))}"
                    st.write(f"**{UI_STRINGS['leaf_analysis']}:** {translate_text(leaf_analysis_text, lang_code)}")
                    st.write(f"**{UI_STRINGS['leaf_disease']}:** {translate_text(result.get('leaf_disease', 'Unknown'), lang_code)}")
                    
                    st.subheader(UI_STRINGS["care_advice"])
                    advice_text = result.get('translated_advice', translate_text('No advice available', lang_code))
                    st.markdown(advice_text)

                    advice_audio_filepath = os.path.join(AUDIO_FOLDER, f"advice_{current_analysis['analysis_id']}.mp3")
                    if st.button(UI_STRINGS["play_advice_button"], key="play_advice"):
                        with st.spinner("Generating audio..."):
                            try:
                                if text_to_audio(advice_text, advice_audio_filepath, lang_code=lang_code):
                                    st.audio(advice_audio_filepath, format='audio/mp3')
                                    st.caption(UI_STRINGS["audio_caption"].format(lang=st.session_state.get('selected_language', 'English')))
                                else:
                                    st.error(UI_STRINGS["audio_failed"])
                            except Exception as e:
                                logger.error(f"Error in audio generation: {e}")
                                st.error(UI_STRINGS["audio_failed"])

                except Exception as e:
                    logger.error(f"Error displaying result: {e}")
                    st.error(UI_STRINGS["app_error"])

            except Exception as e:
                logger.error(f"Error displaying current analysis: {e}")
                st.error(UI_STRINGS["app_error"])

        st.header(UI_STRINGS["follow_up_header"])
        try:
            if 'current_analysis' in st.session_state:
                question = st.text_area(UI_STRINGS["follow_up_placeholder"], key="follow_up_question")
                if st.button(UI_STRINGS["submit_question_button"]):
                    try:
                        if not question.strip():
                            st.warning(UI_STRINGS["no_question_warning"])
                        elif qa_chain is None:
                            st.error(UI_STRINGS["ai_unavailable"])
                        else:
                            with st.spinner("Processing your question..."):
                                try:
                                    conversation_context = build_conversation_context(st.session_state.current_analysis, lang_code)
                                    current_analysis = st.session_state.current_analysis
                                    weather_context = ""
                                    if 'result' in current_analysis and 'weather_data' in current_analysis['result']:
                                        location = current_analysis.get('location', '')
                                        weather_context = format_weather_for_ai(current_analysis['result']['weather_data'], location, lang_code)
                                    
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
                                        st.warning(UI_STRINGS["history_save_warning"])
                                    
                                except Exception as e:
                                    logger.error(f"Error processing question: {e}")
                                    st.error(UI_STRINGS["question_error"])

                    except Exception as e:
                        logger.error(f"Error in question submission: {e}")
                        st.error(UI_STRINGS["question_error"])

                try:
                    if st.session_state.current_analysis.get('questions'):
                        st.subheader(UI_STRINGS["question_history"])
                        for i, q in enumerate(st.session_state.current_analysis['questions']):
                            try:
                                st.write(f"**Q{i+1}:** {q.get('question', 'Question not available')}")
                                answer_text = q.get('translated_answer', translate_text('Answer not available', lang_code))
                                st.write(f"**A{i+1}:** {answer_text}")
                                
                                question_audio_filepath = os.path.join(AUDIO_FOLDER, f"question_{current_analysis['analysis_id']}_{i}.mp3")
                                if st.button(f"Play Answer {i+1}", key=f"play_question_{i}"):
                                    with st.spinner("Generating audio..."):
                                        try:
                                            if text_to_audio(answer_text, question_audio_filepath, lang_code=lang_code):
                                                st.audio(question_audio_filepath, format='audio/mp3')
                                                st.caption(UI_STRINGS["audio_caption"].format(lang=st.session_state.get('selected_language', 'English')))
                                            else:
                                                st.error(UI_STRINGS["audio_failed"])
                                        except Exception as e:
                                            logger.error(f"Error generating question audio: {e}")
                                            st.error(UI_STRINGS["audio_failed"])
                                st.write("---")
                            except Exception as e:
                                logger.error(f"Error displaying question {i}: {e}")
                                st.error(f"Error displaying question {i+1}")
                except Exception as e:
                    logger.error(f"Error displaying question history: {e}")
                    st.error(UI_STRINGS["app_error"])
            else:
                st.info(UI_STRINGS["no_analysis_info"])
        except Exception as e:
            logger.error(f"Error in follow-up questions section: {e}")
            st.error(UI_STRINGS["app_error"])

    except Exception as e:
        logger.error(f"Critical application error: {e}")
        logger.error(traceback.format_exc())
        st.error(UI_STRINGS["app_error"])

    try:
        st.markdown("---")
        st.markdown(f"### {UI_STRINGS['tips_header']}")
        st.markdown(UI_STRINGS["tips_content"])
    except Exception as e:
        logger.error(f"Error displaying footer: {e}")

except Exception as e:
    logger.error(f"Critical application error: {e}")
    logger.error(traceback.format_exc())
    st.error(UI_STRINGS.get("app_error", "Application encountered a critical error. Please refresh the page."))