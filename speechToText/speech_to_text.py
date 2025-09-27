import sys
import os
import speech_recognition as sr
from pydub import AudioSegment
import traceback

def prepare_voice_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        return path
    elif ext in ('.mp3', '.m4a', '.ogg', '.flac', '.webm'):
        audio_file = AudioSegment.from_file(path, format=ext[1:])
        wav_file = os.path.splitext(path)[0] + '.wav'
        audio_file.export(wav_file, format='wav')
        return wav_file
    else:
        raise ValueError(f'Unsupported audio format: {ext}')

def transcribe_audio(audio_data, language) -> str:
    r = sr.Recognizer()
    return r.recognize_google(audio_data, language=language)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("")
        sys.exit(1)

    input_path = sys.argv[1]
    language = sys.argv[2]

    try:
        wav_file = prepare_voice_file(input_path)
        with sr.AudioFile(wav_file) as source:
            audio_data = sr.Recognizer().record(source)
            try:
                text = transcribe_audio(audio_data, language)
                print(text.strip())
            except sr.UnknownValueError:
                print("", file=sys.stderr)
                sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
