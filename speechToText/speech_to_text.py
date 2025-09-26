import os
import speech_recognition as sr
from pydub import AudioSegment

def prepare_voice_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        return path
    elif ext in ('.mp3', '.m4a', '.ogg', '.flac'):
        audio_file = AudioSegment.from_file(path, format=ext[1:])
        wav_file = os.path.splitext(path)[0] + '.wav'
        audio_file.export(wav_file, format='wav')
        return wav_file
    else:
        raise ValueError(f'Unsupported audio format: {ext}')

def transcribe_audio(audio_data, language) -> str:
    r = sr.Recognizer()
    return r.recognize_google(audio_data, language=language)

def format_transcription(text: str) -> str:
    text = text.strip().lower()
    if text:
        text = text[0].upper() + text[1:]
    return text

def write_transcription_to_file(text, output_file) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

def get_default_output_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + '.txt'

def speech_to_text(input_path: str, language: str) -> None:
    wav_file = prepare_voice_file(input_path)
    with sr.AudioFile(wav_file) as source:
        audio_data = sr.Recognizer().record(source)
        text = transcribe_audio(audio_data, language)
        formatted_text = format_transcription(text)
        output_path = get_default_output_path(input_path)
        write_transcription_to_file(formatted_text, output_path)
        print('Transcription:')
        print(formatted_text)
        print(f'Transcription saved to: {output_path}')

if __name__ == '__main__':
    print('Please enter the path to an audio file (WAV, MP3, M4A, OGG, or FLAC):')
    input_path = input().strip()
    if not os.path.isfile(input_path):
        print('Error: File not found.')
        exit(1)
    print('Please enter the language code (e.g. en-US):')
    language = input().strip()
    try:
        speech_to_text(input_path, language)
    except Exception as e:
        print('Error:', e)
        exit(1)