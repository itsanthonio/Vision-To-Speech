import logging
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
from TTS.api import TTS
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the TTS model for Twi
tts_model = TTS(model_name="tts_models/tw_asante/openbible/vits")

# Define the function to generate English captions
def generate_english_caption(image_file):
    try:
        image = Image.open(image_file)
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logging.error(f"Failed to generate English caption: {e}")
        return None

# Define the function to translate text to Twi
def translate(text, target_language):
    try:
        ngrok_url = "https://9482-41-79-97-5.ngrok-free.app"
        url = f"{ngrok_url}/translate"
        headers = {"Content-Type": "application/json"}
        data = {"text": text, "to": target_language}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("translatedText", "Translation result not found")
    except requests.RequestException as e:
        logging.error(f"Translation request failed: {e}")
        return "Translation error"

# Define the function to generate and save Twi audio
def save_audio(twi_caption, audio_path):
    try:
        tts_model.tts_to_file(text=twi_caption, file_path=audio_path)
    except Exception as e:
        logging.error(f"Failed to save audio file: {e}")

# Streamlit code for uploading and processing the image
st.title("VerMa Captioning and Translation Webpage")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Generate English caption
    english_caption = generate_english_caption(uploaded_file)
    if english_caption:
        st.write("English Caption:", english_caption)

        # Translate English caption to Twi
        twi_caption = translate(english_caption, target_language="tw")
        st.write("Twi Caption:", twi_caption)

        # Generate Twi audio
        audio_path = f"{os.path.splitext(uploaded_file.name)[0]}_twi.wav"
        save_audio(twi_caption, audio_path)
        st.write(f"Audio saved at {audio_path}")

        # Display the audio player
        st.audio(audio_path)
    else:
        st.write("Failed to generate English caption.")
