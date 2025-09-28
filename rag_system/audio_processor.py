import math

import assemblyai as aai
from pydub import AudioSegment

from constants import ASEEMBLY_API_KEY

# Set your API key
aai.settings.api_key = ASEEMBLY_API_KEY

def transcribe_audio_segment(audio_file: str, start_sec: float, end_sec: float):
    """
    Transcribe a specific segment of an audio file using AssemblyAI
    """
    try:
        # Create a temporary file for the audio segment
        audio = AudioSegment.from_file(audio_file)
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        segment = audio[start_ms:end_ms]
        
        # Export segment to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            segment.export(temp_file.name, format="mp3")
            temp_path = temp_file.name
        
        # Transcribe the segment
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
        transcript = aai.Transcriber(config=config).transcribe(temp_path)
        
        # Clean up temporary file
        import os
        os.unlink(temp_path)
        
        if transcript.status == "error":
            return f"Transcription unavailable for segment {start_sec:.1f}s - {end_sec:.1f}s"
        
        return transcript.text if transcript.text else f"Audio content from {start_sec:.1f}s to {end_sec:.1f}s"
        
    except Exception as e:
        return f"Audio content from {start_sec:.1f}s to {end_sec:.1f}s"

def transcribe_audio_with_timestamps(audio_file: str):
    """
    Transcribe entire audio file with word-level timestamps for more accurate chunking
    """
    try:
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.universal,
            word_timestamps=True
        )
        transcript = aai.Transcriber(config=config).transcribe(audio_file)

        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription with timestamps failed: {e}")

def create_audio_chunks(audio_file: str, chunk_duration: int = None):
    """
    Create optimal audio chunks with actual transcription content
    """
    try:
        # Get audio duration
        audio = AudioSegment.from_file(audio_file)
        total_duration_ms = len(audio)
        total_duration_sec = total_duration_ms / 1000.0
        
        # Determine optimal chunk size based on total duration
        chunk_duration = calculate_optimal_chunk_size(total_duration_sec)
        
        # First, try to get transcription with word-level timestamps for more accuracy
        try:
            transcript = transcribe_audio_with_timestamps(audio_file)
            return create_chunks_with_timestamps(transcript, total_duration_sec, chunk_duration, audio_file)
        except Exception as e:
            print(f"Using fallback chunking: {e}")
            return create_chunks_with_segment_transcription(audio_file, total_duration_sec, chunk_duration)
        
    except Exception as e:
        raise RuntimeError(f"Audio chunking failed: {e}")

def create_chunks_with_timestamps(transcript, total_duration_sec: float, chunk_duration: float, audio_file: str):
    """
    Create chunks using word-level timestamps for accurate text alignment
    """
    chunks = []
    current_chunk_start = 0
    
    while current_chunk_start < total_duration_sec:
        chunk_end = min(current_chunk_start + chunk_duration, total_duration_sec)
        chunk_num = len(chunks) + 1
        total_chunks = math.ceil(total_duration_sec / chunk_duration)
        
        # Extract words that fall within this chunk
        chunk_words = []
        if hasattr(transcript, 'words') and transcript.words:
            for word in transcript.words:
                word_start = word.start / 1000.0  # Convert ms to seconds
                if word_start >= current_chunk_start and word_start < chunk_end:
                    chunk_words.append(word.text)
        
        if chunk_words:
            content = ' '.join(chunk_words)
        else:
            # Fallback to segment transcription if no word timestamps
            content = transcribe_audio_segment(audio_file, current_chunk_start, chunk_end)
        
        chunks.append({
            'start_sec': current_chunk_start,
            'end_sec': chunk_end,
            'duration': chunk_end - current_chunk_start,
            'content': content,
            'chunk_index': len(chunks),
            'total_chunks': total_chunks
        })
        
        current_chunk_start = chunk_end
    
    return chunks

def create_chunks_with_segment_transcription(audio_file: str, total_duration_sec: float, chunk_duration: float):
    """
    Fallback method: transcribe each segment individually
    """
    chunks = []
    current_chunk_start = 0
    
    while current_chunk_start < total_duration_sec:
        chunk_end = min(current_chunk_start + chunk_duration, total_duration_sec)
        chunk_num = len(chunks) + 1
        total_chunks = math.ceil(total_duration_sec / chunk_duration)
        
        # Transcribe this specific segment
        content = transcribe_audio_segment(audio_file, current_chunk_start, chunk_end)
        
        chunks.append({
            'start_sec': current_chunk_start,
            'end_sec': chunk_end,
            'duration': chunk_end - current_chunk_start,
            'content': content,
            'chunk_index': len(chunks),
            'total_chunks': total_chunks
        })
        
        current_chunk_start = chunk_end
    
    return chunks

def calculate_optimal_chunk_size(total_duration_sec: float) -> int:
    """
    Calculate optimal chunk size based on total audio duration
    """
    # Rule-based chunk sizing
    if total_duration_sec <= 60:  # 1 minute or less
        # For very short audio, use 4 chunks
        return 30
    
    elif total_duration_sec <= 300:  # 5 minutes or less
        # For short audio, use 12 chunks (like 1/12th)
        return 150
    
    elif total_duration_sec <= 900:  # 15 minutes or less
        # For medium audio, use 30-second chunks
        return 200
    
    elif total_duration_sec <= 1800:  # 30 minutes or less
        # For longer audio, use 45-second chunks
        return 250
    
    else:  # More than 30 minutes
        # For very long audio, use 60-second chunks
        return 300

def create_fixed_chunks(audio_file: str, num_chunks: int = 12):
    """
    Alternative: Create exactly N chunks with actual transcription
    """
    try:
        audio = AudioSegment.from_file(audio_file)
        total_duration_ms = len(audio)
        total_duration_sec = total_duration_ms / 1000.0
        
        chunk_duration = total_duration_sec / num_chunks
        
        chunks = []
        current_chunk_start = 0
        
        for i in range(num_chunks):
            chunk_end = min(current_chunk_start + chunk_duration, total_duration_sec)
            
            # For the last chunk, ensure we cover the entire duration
            if i == num_chunks - 1:
                chunk_end = total_duration_sec
            
            # Transcribe this specific segment
            content = transcribe_audio_segment(audio_file, current_chunk_start, chunk_end)
            
            chunks.append({
                'start_sec': current_chunk_start,
                'end_sec': chunk_end,
                'duration': chunk_end - current_chunk_start,
                'content': content,
                'chunk_index': i,
                'total_chunks': num_chunks
            })
            
            current_chunk_start = chunk_end
            
        return chunks
        
    except Exception as e:
        raise RuntimeError(f"Fixed chunking failed: {e}")

def transcribe_audio(audio_file: str):
    """
    Original transcription function for backward compatibility
    """
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
    transcript = aai.Transcriber(config=config).transcribe(audio_file)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript.text