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