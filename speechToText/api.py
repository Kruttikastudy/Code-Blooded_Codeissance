from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import uuid
import io

app = Flask(__name__)
CORS(app)

def transcribe_audio_from_bytes(audio_bytes, language='en-US') -> str:
    """Transcribe audio from bytes directly"""
    try:
        # Create recognizer
        r = sr.Recognizer()
        
        # Convert bytes to audio file-like object
        audio_io = io.BytesIO(audio_bytes)
        
        # Use pydub to convert to wav format
        audio_segment = AudioSegment.from_file(audio_io, format="webm")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Use speech recognition
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language=language)
            return text.strip()
            
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service; {e}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en-US')
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read audio file as bytes
        audio_bytes = audio_file.read()
        
        if len(audio_bytes) == 0:
            return jsonify({'error': 'Empty audio file'}), 400
        
        # Transcribe directly from bytes
        transcript = transcribe_audio_from_bytes(audio_bytes, language)
        
        if transcript.startswith("Could not") or transcript.startswith("Error"):
            return jsonify({'error': transcript, 'success': False}), 400
        
        # Save to output file
        output_path = f"output_{uuid.uuid4().hex[:8]}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        return jsonify({
            'transcript': transcript,
            'output_file': output_path,
            'success': True
        })
                
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}', 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
