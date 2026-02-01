import tempfile
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq()

def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    with open(temp_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3"
        )

    return transcription.text
