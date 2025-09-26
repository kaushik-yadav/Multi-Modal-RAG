import assemblyai as aai

from constants import ASEEMBLY_API_KEY

# Set your API key
aai.settings.api_key = ASEEMBLY_API_KEY

def transcribe_audio(audio_file: str) -> str:
    """
    Transcribe audio file using AssemblyAI
    """
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
    transcript = aai.Transcriber(config=config).transcribe(audio_file)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript.text

