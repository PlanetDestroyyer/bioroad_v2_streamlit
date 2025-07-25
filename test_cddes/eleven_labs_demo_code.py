# from elevenlabs import Voice, VoiceSettings, generate
# from dotenv import load_dotenv
# from pydub import AudioSegment
# from pydub.playback import play
# import os
# import io

# # Load API key
# load_dotenv()
# api_key = os.getenv("ELEVEN_LABS_API_KEY")


# text ="""
# सुरुवातीच्या फळाच्या टप्प्यावर, आपल्या केळीच्या वनस्पतीला उच्च पोटॅशियम आणि फॉस्फरस आवश्यक आहे.फळे पूर्ण आकारात असतील परंतु तरीही हिरव्या असतील.आपल्याला हे समजेल की जेव्हा फळांचा ओघ अदृश्य होतो, बोटांनी भरुन काढले जाते आणि रंग गडद ते हलका हिरव्या रंगात बदलतो तेव्हा कापणी करण्याची वेळ आली आहे.वारा संरक्षण आवश्यक आहे."""
# # Generate audio from ElevenLabs
# audio = generate(
#     model="eleven_multilingual_v2",
#     api_key=api_key,
#     text=text,
#     voice=Voice(
#         voice_id="4U2MtPm7Mj91nh3AIC1V",

#         settings=VoiceSettings(
#             stability=0.1,
#             similarity_boost=0.5,
#             style=0.0,
#             use_speaker_boost=True,
            
#         )
#     )
#     )
# audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
# play(audio_segment)

from elevenlabs import Voice, VoiceSettings, generate
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
import os
import io

# Load API key
load_dotenv()
api_key = os.getenv("ELEVEN_LABS_API_KEY")

# SSML text with slower speed
text = """

  आपल्या विश्लेषणाच्या आधारे, आपल्या 3 महिन्यांच्या केळीची वनस्पती, केळीची वनस्पती, बीपासून नुकतेच तयार झालेले अवस्था आहे आणि त्याला केळी कॉर्डाना नावाचा एक पान रोग आहे. येथे काही काळजी सल्लाः केळी कॉर्डानाला संबोधित करा: केळी कॉर्डानासाठी संशोधन उपचार. आपल्याला बुरशीनाशक वापरण्याची आवश्यकता असू शकते. बुरशीनाशकावरील सूचनांचे काळजीपूर्वक अनुसरण करा

"""

# Generate audio with SSML
audio = generate(
    api_key=api_key,
    text=text,
    model="eleven_multilingual_v2",  # required for SSML
    voice=Voice(
        voice_id="TX3LPaxmHKxFdv7VOQHJ",
        settings=VoiceSettings(
            stability=0.71,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True
        )
    ),
    stream=False  # needed for SSML text
)

# Convert byte stream to AudioSegment and play
audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
play(audio_segment)
