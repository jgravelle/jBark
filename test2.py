import os
import time
import numpy as np
from jbark import JBark, SAMPLE_RATE
from scipy.io import wavfile
import librosa

class ExpandedJBarkTestSuite:
    def __init__(self):
        print("Initializing expanded jBark test suite...")
        self.jbark = JBark()
        self.test_output_dir = "expanded_test_outputs"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def run_tests(self):
        self.test_audio_generation_variations()
        self.test_voice_cloning_variations()
        self.test_voice_conversion_variations()
        self.test_language_support()
        self.test_error_handling()
        self.test_performance()

    def test_audio_generation_variations(self):
        print("\nTesting audio generation variations...")
        
        # Test with different text lengths
        texts = [
            "Short text.",
            "This is a medium-length text with some variation in words and punctuation!",
            "This is a longer text that includes multiple sentences. It has various punctuation marks, like commas, periods, and even a question mark? The goal is to test how jBark handles longer inputs with more complexity."
        ]
        
        for i, text in enumerate(texts):
            output_path = os.path.join(self.test_output_dir, f"test_length_{i}.wav")
            audio = self.jbark.generate_audio(text, output_path)
            print(f"Generated audio for text length {i+1}. Shape: {audio.shape}")
        
        # Test with different history prompts
        history_prompts = [None, "v2/en_speaker_6", "v2/en_speaker_9"]
        text = "This is a test of different voice presets."
        
        for i, prompt in enumerate(history_prompts):
            output_path = os.path.join(self.test_output_dir, f"test_prompt_{i}.wav")
            audio = self.jbark.generate_audio(text, output_path, history_prompt=prompt)
            print(f"Generated audio with history prompt {prompt}. Shape: {audio.shape}")

    def test_voice_cloning_variations(self):
        print("\nTesting voice cloning variations...")
        
        # Generate sample voices with different characteristics
        sample_texts = [
            "This is a sample with a deep voice.",
            "This is a sample with a high-pitched voice.",
            "This is a sample with a fast-paced voice.",
            "This is a sample with a slow-paced voice."
        ]
        
        for i, text in enumerate(sample_texts):
            sample_path = os.path.join(self.test_output_dir, f"sample_voice_{i}.wav")
            self.jbark.generate_audio(text, sample_path)
            
            voice_chars = self.jbark.simple_voice_clone(sample_path)
            print(f"Voice characteristics for sample {i}: {voice_chars}")
            
            # Test cloning with extracted characteristics
            clone_text = "This is a test of voice cloning."
            clone_output = os.path.join(self.test_output_dir, f"cloned_voice_{i}.wav")
            cloned_audio = self.jbark.generate_with_cloned_voice(clone_text, voice_chars, clone_output)
            print(f"Generated cloned audio for sample {i}. Shape: {cloned_audio.shape}")

    def test_voice_conversion_variations(self):
        print("\nTesting voice conversion variations...")
        
        base_text = "This is the base voice for conversion tests."
        base_audio_path = os.path.join(self.test_output_dir, "base_voice.wav")
        base_audio = self.jbark.generate_audio(base_text, base_audio_path)
        
        # Test with different pitch and tempo combinations
        variations = [
            {"pitch": -0.5, "tempo": 80},
            {"pitch": 0.5, "tempo": 160},
            {"pitch": 0, "tempo": 120},
            {"pitch": 1, "tempo": 100},
            {"pitch": -1, "tempo": 140}
        ]
        
        for i, vars in enumerate(variations):
            converted_audio = self.jbark.simple_voice_conversion(base_audio, vars)
            output_path = os.path.join(self.test_output_dir, f"converted_voice_{i}.wav")
            wavfile.write(output_path, SAMPLE_RATE, converted_audio)
            print(f"Generated converted audio with pitch {vars['pitch']} and tempo {vars['tempo']}. Shape: {converted_audio.shape}")

    def test_language_support(self):
        print("\nTesting language support...")
        
        languages = self.jbark.list_supported_languages()
        test_text = "Hello, this is a test."
        
        for lang_code, lang_name in languages.items():
            output_path = os.path.join(self.test_output_dir, f"test_{lang_code}.wav")
            try:
                audio = self.jbark.generate_audio(test_text, output_path, history_prompt=f"v2/{lang_code}_speaker_1")
                print(f"Generated audio for {lang_name} ({lang_code}). Shape: {audio.shape}")
            except Exception as e:
                print(f"Error generating audio for {lang_name} ({lang_code}): {str(e)}")

    def test_error_handling(self):
        print("\nTesting error handling...")
        
        # Test with empty text
        try:
            self.jbark.generate_audio("")
        except Exception as e:
            print(f"Empty text error handled: {str(e)}")
        
        # Test with invalid history prompt
        try:
            self.jbark.generate_audio("Test", history_prompt="invalid_prompt")
        except Exception as e:
            print(f"Invalid history prompt error handled: {str(e)}")
        
        # Test with invalid audio path for voice cloning
        try:
            self.jbark.simple_voice_clone("non_existent_audio.wav")
        except Exception as e:
            print(f"Invalid audio path error handled: {str(e)}")

    def test_performance(self):
        print("\nTesting performance...")
        
        text_lengths = [10, 50, 100, 200, 500]
        
        for length in text_lengths:
            text = " ".join(["test"] * length)
            start_time = time.time()
            audio = self.jbark.generate_audio(text)
            end_time = time.time()
            print(f"Generated audio for text length {length}. Time taken: {end_time - start_time:.2f} seconds. Audio shape: {audio.shape}")

if __name__ == "__main__":
    test_suite = ExpandedJBarkTestSuite()
    test_suite.run_tests()