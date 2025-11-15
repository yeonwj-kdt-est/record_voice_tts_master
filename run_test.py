from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,
    SUPPORTED_LANGUAGES
)

if __name__ == "__main__":
    try:
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    kr_text = ' 밖에 비가 많이오는데 집에 어떻게 가지?'
    kr_wav = MODEL.generate(kr_text, language_id='ko')
    print(len(kr_wav))
