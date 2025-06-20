import os # For file and directory operations
import re # For regular expressions (used in text cleaning)

import numpy as np # For numerical operations, especially array concatenation
import soundfile as sf # For reading and writing audio files
import streamlit as st # For building the web UI
from kokoro import KPipeline # For text-to-speech synthesis
from langchain_core.prompts import ChatPromptTemplate # For prompt templating
from langchain_ollama import ChatOllama # For LLM chat interface

# Directory where generated audio files will be stored
audios_directory = 'audios/'

# Dictionary mapping language names (with flag emojis) to language codes for TTS
supported_languages = {
    'American English': 'a',
    'British English': 'b',
    'Spanish': 'e',
    'French': 'f',
    'Italian': 'i',
    'Japanese': 'j',
    'Brazilian Portuguese': 'p',
    'Mandarin Chinese': 'z'
}

# Prompt template for summarizing text using the LLM
summary_template = """
Summarize the following text by highlighting the key points.
Maintain a conversational tone and keep the summary easy to follow for a general audience.
Text: {text}
"""

# Initialize the LLM model for summarization
model = ChatOllama(model="deepseek-r1:8b")

def generate_audio(pipeline, text):
    """
    Generate an audio file from the given text using the provided TTS pipeline.

    Steps:
    1. Remove any existing audio files in the output directory to avoid clutter.
    2. Use the pipeline to synthesize audio from the text, collecting audio chunks.
    3. Concatenate all audio chunks into a single audio array.
    4. Write the audio array to a .wav file in the output directory.
    5. Return the name of the generated audio file.

    Args:
        pipeline: An instance of KPipeline configured for the selected language.
        text (str): The text to convert to speech.

    Returns:
        str: The filename of the generated audio file.
    """
    # Remove all existing files in the audio directory
    for file in os.listdir(audios_directory):
        os.remove(os.path.join(audios_directory, file))

    # Generate audio chunks from the text using the pipeline
    generator = pipeline(text, voice='af_heart')
    chunks = []

    # Collect all audio chunks produced by the generator
    for i, (gs, ps, audio) in enumerate(generator):
        chunks.append(audio)

    file_name = 'audio.wav'
    # Concatenate all audio chunks into a single array
    full_audio = np.concatenate(chunks, axis=0)
    # Write the audio array to a .wav file at 24kHz sample rate
    sf.write(os.path.join(audios_directory, file_name), full_audio, 24000)

    return file_name

def summarize_text(text):
    """
    Summarize the input text using the LLM and the summary template.

    Steps:
    1. Format the prompt with the input text.
    2. Invoke the LLM to generate a summary.
    3. Clean the summary output to remove unwanted tags.
    4. Return the cleaned summary.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summarized and cleaned text.
    """
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model

    summary = chain.invoke({"text": text})
    return clean_text(summary.content)

def clean_text(text):
    """
    Remove any <think>...</think> tags and their contents from the text.
    Strips leading/trailing whitespace.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# --- Streamlit UI Section ---

# Set the title of the web app
st.title("AI Podcaster")

# Dropdown for language selection
language = st.selectbox("Select a language:", list(supported_languages.keys()), index=0)

# Text area for user to input the text to be converted to audio
text = st.text_area("Enter text to generate audio:")

# Checkbox to optionally summarize the input text before audio generation
should_summarize = st.checkbox("Summarize text")

# Button to trigger audio generation
button = st.button("Generate Audio")

# Main logic: when the button is pressed and text is provided
if button and text:
    # Initialize the TTS pipeline with the selected language code
    pipeline = KPipeline(lang_code=supported_languages[language])

    # If summarization is requested, summarize the input text first
    if should_summarize:
        text = summarize_text(text)

    # Generate the audio file from (possibly summarized) text
    file_name = generate_audio(pipeline, text)
    # Play the generated audio in the Streamlit app
    st.audio(os.path.join(audios_directory, file_name))