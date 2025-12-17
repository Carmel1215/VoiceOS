from core.whisper_manager import *
from core.executor import *


def main():
    model = load_model()
    print('설명문')
    while True:
        input()
        audio = record_audio()
        print('transcribing...')
        text = transcribe(model, audio)
        print('>>>', text if text else 'None')
        execute(text.replace(' ', ''))

if __name__ == "__main__":
    main()