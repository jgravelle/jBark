# jbark.py

```python
import numpy as np
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io import wavfile
import librosa
import resampy
import warnings

class JBark:
    def __init__(self):
        print("Using CPU for computations.")
        self._suppress_warnings()
        self.preload_models()

    def _suppress_warnings(self):
        # Suppress all FutureWarnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Suppress the specific warning about torch.load
        warnings.filterwarnings("ignore", message="You are using `torch.load`")
        
        # Suppress the warning about clean_up_tokenization_spaces
        warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
        
        # Suppress the warning about torch.nn.utils.weight_norm
        warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

    def preload_models(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preload_models()

    def generate_audio(self, text_prompt, output_path=None, history_prompt=None):
        """
        Generate audio from text using Bark.
        
        :param text_prompt: The text to convert to speech
        :param output_path: Path to save the generated audio (optional)
        :param history_prompt: Voice preset to use (optional)
        :return: Audio array
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_array = generate_audio(text_prompt, history_prompt=history_prompt)
        
        if output_path:
            wavfile.write(output_path, SAMPLE_RATE, audio_array)
        
        return audio_array

    def simple_voice_clone(self, audio_path):
        """
        Analyze an audio sample to extract basic voice characteristics.
        
        :param audio_path: Path to the audio sample
        :return: Dictionary containing basic voice characteristics
        """
        y, sr = librosa.load(audio_path, sr=None)
        pitch = librosa.pitch_tuning(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            "pitch": pitch,
            "tempo": tempo
        }

    def generate_with_cloned_voice(self, text_prompt, voice_characteristics, output_path):
        """
        Generate audio using simple voice conversion.
        
        :param text_prompt: The text to convert to speech
        :param voice_characteristics: Dictionary containing voice characteristics
        :param output_path: Path to save the generated audio
        :return: Audio array
        """
        # Generate base audio using Bark
        base_audio = self.generate_audio(text_prompt)
        
        # Apply simple voice conversion
        converted_audio = self.simple_voice_conversion(base_audio, voice_characteristics)
        
        # Save the converted audio
        wavfile.write(output_path, SAMPLE_RATE, converted_audio)
        
        return converted_audio

    def simple_voice_conversion(self, audio, voice_characteristics):
        """
        Apply simple voice conversion based on pitch and tempo.
        
        :param audio: Input audio array
        :param voice_characteristics: Dictionary containing voice characteristics
        :return: Converted audio array
        """
        # Extract pitch and tempo from voice characteristics
        target_pitch = voice_characteristics.get("pitch", 0)
        target_tempo = voice_characteristics.get("tempo", 120)
        
        # Convert audio to floating point and normalize
        audio_float = librosa.util.normalize(audio.astype(np.float32))
        
        # Pitch shift
        n_steps = target_pitch * 12  # Convert octaves to semitones
        audio_pitched = librosa.effects.pitch_shift(audio_float, sr=SAMPLE_RATE, n_steps=n_steps)
        
        # Custom time stretch
        rate = target_tempo / 120  # Tempo adjustment factor
        audio_stretched = self.custom_time_stretch(audio_pitched, rate)
        
        # Ensure the output has the same length as the input
        if len(audio_stretched) > len(audio):
            audio_converted = audio_stretched[:len(audio)]
        elif len(audio_stretched) < len(audio):
            audio_converted = np.pad(audio_stretched, (0, len(audio) - len(audio_stretched)))
        else:
            audio_converted = audio_stretched
        
        # Convert back to int16
        audio_converted = (audio_converted * 32767).astype(np.int16)
        
        return audio_converted

    def custom_time_stretch(self, audio, rate):
        """
        Custom time stretching function using resampling.
        
        :param audio: Input audio array
        :param rate: Time stretch rate
        :return: Time-stretched audio array
        """
        new_length = int(len(audio) / rate)
        return resampy.resample(audio, len(audio), new_length)

    @staticmethod
    def list_supported_languages():
        """
        List all supported languages by Bark.
        
        :return: Dictionary of supported languages
        """
        return {
            "en": "English",
            "de": "German",
            "es": "Spanish",
            "fr": "French",
            "hi": "Hindi",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "pl": "Polish",
            "pt": "Portuguese",
            "ru": "Russian",
            "tr": "Turkish",
            "zh": "Chinese (Simplified)"
        }

# Example usage
if __name__ == "__main__":
    jbark = JBark()
    
    # Generate audio using default Bark
    audio = jbark.generate_audio("Hello, this is a test.", output_path="test_output.wav")
    
    # Analyze a voice sample
    voice_chars = jbark.simple_voice_clone("sample_voice.wav")
    
    # Generate audio using the analyzed voice characteristics
    cloned_audio = jbark.generate_with_cloned_voice(
        "This is a test with a cloned voice.", 
        voice_chars, 
        "cloned_voice_output.wav"
    )
    
    # List supported languages
    languages = jbark.list_supported_languages()
    print("Supported languages:", languages)
```

# test.py

```python
import torch
print(torch.cuda.is_available())
```

# test_jbark.py

```python
# test_jbark.py

import os
import time
from jbark import JBark

class JBarkTestSuite:
    def __init__(self):
        print("Initializing jBark test suite...")
        self.jbark = JBark()
        self.test_output_dir = "test_outputs"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def run_tests(self):
        while True:
            self.display_menu()
            choice = input("Enter your choice (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            self.execute_test(choice)

    def display_menu(self):
        print("\n--- jBark Test Suite ---")
        print("1. Test audio generation")
        print("2. Test simple voice cloning")
        print("3. Test generation with simple voice conversion")
        print("4. Test supported languages")
        print("5. Test all functions")
        print("q. Quit")

    def execute_test(self, choice):
        if choice == '1':
            self.test_audio_generation()
        elif choice == '2':
            self.test_simple_voice_clone()
        elif choice == '3':
            self.test_generation_with_simple_voice_conversion()
        elif choice == '4':
            self.test_supported_languages()
        elif choice == '5':
            self.test_all_functions()
        else:
            print("Invalid choice. Please try again.")

    def test_audio_generation(self):
        print("\nTesting audio generation...")
        text_prompt = "Hello, (laugh) this is a test of... um, jBark audio generation."
        output_path = os.path.join(self.test_output_dir, "test_audio_generation.wav")
        
        start_time = time.time()
        audio = self.jbark.generate_audio(text_prompt, output_path)
        end_time = time.time()

        print(f"Audio generated and saved to: {output_path}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print(f"Audio array shape: {audio.shape}")

    def test_simple_voice_clone(self):
        print("\nTesting simple voice cloning...")
        sample_audio = os.path.join(self.test_output_dir, "sample_voice.wav")

        # Generate a sample audio for cloning if it doesn't exist
        if not os.path.exists(sample_audio):
            print("Generating sample audio for cloning...")
            self.jbark.generate_audio("This is a sample voice for cloning.", sample_audio)

        start_time = time.time()
        voice_chars = self.jbark.simple_voice_clone(sample_audio)
        end_time = time.time()

        print(f"Voice characteristics extracted: {voice_chars}")
        print(f"Extraction time: {end_time - start_time:.2f} seconds")

    def test_generation_with_simple_voice_conversion(self):
        print("\nTesting generation with simple voice conversion...")
        sample_audio = os.path.join(self.test_output_dir, "sample_voice.wav")
        output_path = os.path.join(self.test_output_dir, "test_simple_voice_conversion_output.wav")

        if not os.path.exists(sample_audio):
            print("Generating sample audio for voice conversion...")
            self.jbark.generate_audio("This is a sample voice for conversion.", sample_audio)

        voice_chars = self.jbark.simple_voice_clone(sample_audio)
        text_prompt = "This is a test of generating audio with simple voice conversion."
        
        start_time = time.time()
        audio = self.jbark.generate_with_cloned_voice(text_prompt, voice_chars, output_path)
        end_time = time.time()

        print(f"Audio generated with simple voice conversion and saved to: {output_path}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print(f"Audio array shape: {audio.shape}")

    def test_supported_languages(self):
        print("\nTesting supported languages...")
        languages = self.jbark.list_supported_languages()
        print("Supported languages:")
        for code, name in languages.items():
            print(f"  {code}: {name}")

    def test_all_functions(self):
        print("\nRunning all tests...")
        self.test_audio_generation()
        self.test_simple_voice_clone()
        self.test_generation_with_simple_voice_conversion()
        self.test_supported_languages()

if __name__ == "__main__":
    test_suite = JBarkTestSuite()
    test_suite.run_tests()
```

