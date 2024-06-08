from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks
import openai
import requests


# Function to clean and format the passenger ID
def format_passenger_id(passenger_id):
    # Remove any dashes or spaces
    cleaned_id = passenger_id.replace('-', '').replace(' ', '')
    # Format as XXXX XXXXXX
    formatted_id = f"{cleaned_id[:4]} {cleaned_id[4:]}"
    return formatted_id


# Function to download the audio file from a URL
def download_audio(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception("Failed to download the audio file.")


# Function to extract text from audio using OpenAI's API
def extract_text_from_audio(audio_url):
    client = OpenAI(api_key='...')
    if audio_url.startswith('http'):
        audio_path = '/content/downloaded_audio.mp3'  # Set the destination path
        success = download_audio(audio_url, audio_path)
        if not success:
            print("Failed to download the audio file.")
            return None
    else:
        # Otherwise, assume it's a local file path
        audio_path = audio_url

    with open(audio_path, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file= audio_file,
            response_format="text")

    transcript = transcript.replace('\n', '').strip()
    print(transcript)
    return format_passenger_id(transcript)
