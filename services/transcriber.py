import os
import json
from difflib import SequenceMatcher

class SpeechTranscriber:
    _instance = None

    def __init__(self):
        self.model_name = "facebook/wav2vec2-base-960h"
        self.tokenizer = None
        self.model = None
        self.device = None
        self.phonetic_data = self._load_phonetic_data()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SpeechTranscriber()
        return cls._instance

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Transcriber] Loading {self.model_name} onto {self.device}...")
            self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def _load_phonetic_data(self):
        json_path = os.path.join(os.path.dirname(__file__), "..", "data", "phonetic.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load phonetic.json: {e}")
        return {}

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe the audio file using Wav2Vec2.
        """
        import torch
        import librosa
        self._load_model()
        audio, _ = librosa.load(audio_path, sr=16000)
        input_values = self.tokenizer(audio, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.decode(predicted_ids[0])
        return transcription.strip().lower()

    def compare_texts(self, target_text: str, user_transcription: str):
        """
        Compare target text and user transcription to find mismatched characters/words,
        and generate structured alignment data for the frontend.
        """
        target_clean = target_text.strip().lower()
        user_clean = user_transcription.strip().lower()

        # Find extra/missing characters
        target_chars = set(c for c in target_clean if c.isalpha())
        user_chars = set(c for c in user_clean if c.isalpha())
        
        mismatched_letters = list(target_chars - user_chars)

        # Word-level comparison
        target_words = target_clean.split()
        user_words = user_clean.split()

        matcher = SequenceMatcher(None, target_words, user_words)
        mismatched_words = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'delete'):
                mismatched_words.extend(target_words[i1:i2])

        # Find matching video recommendations from phonetic.json
        video_links = []
        video_id = self.phonetic_data.get("video_id", "wBuA589kfMg")

        checked_phonemes = set(mismatched_letters)
        for phoneme in checked_phonemes:
            if phoneme in self.phonetic_data:
                item = self.phonetic_data[phoneme]
                start_time = item.get("start_time", 0)
                end_time = item.get("end_time", 0)
                video_url = f"https://www.youtube.com/embed/{video_id}?start={start_time}&end={end_time}&autoplay=0"
                video_links.append({
                    "phoneme": phoneme,
                    "url": video_url,
                    "direct_url": f"https://www.youtube.com/watch?v={video_id}&t={start_time}s",
                    "start_time": start_time,
                    "end_time": end_time
                })

        return {
            "mismatched_letters": mismatched_letters,
            "mismatched_words": mismatched_words if mismatched_words else mismatched_letters,
            "video_links": video_links
        }

def transcribe_and_compare(audio_path: str, target_text: str):
    transcriber = SpeechTranscriber.get_instance()
    user_transcription = transcriber.transcribe(audio_path)
    comparison = transcriber.compare_texts(target_text, user_transcription)
    return user_transcription, comparison
