# jBark Library Documentation
![image](https://github.com/user-attachments/assets/b1c6930a-6e6c-4488-95d8-91bc1563b1f8)


## Table of Contents
1. [Overview and Introduction](#overview-and-introduction)
2. [Installation Guide](#installation-guide)
3. [Usage Instructions](#usage-instructions)
4. [Configuration and Customization](#configuration-and-customization)
5. [API Reference](#api-reference)
6. [Code Architecture and Design](#code-architecture-and-design)
7. [Testing and Debugging](#testing-and-debugging)
8. [Common Issues and FAQs](#common-issues-and-faqs)

## Overview and Introduction

jBark is a powerful Python library that builds upon the capabilities of the original Bark text-to-speech project, adding simple voice conversion features. It provides a seamless interface for generating high-quality speech from text, extracting basic voice characteristics, and applying these characteristics to generated audio.

Key features of jBark include:
- Text-to-speech generation using the Bark model
- Simple voice characteristic extraction
- Basic voice conversion using pitch shifting and tempo adjustment
- Support for multiple languages
- CPU-based computations (no GPU required)
- Suppression of common warnings for a cleaner user experience

Whether you're developing a virtual assistant, creating audiobooks, or working on any project that requires flexible and high-quality speech synthesis, jBark provides the tools you need to bring your ideas to life.

## Installation Guide

To install jBark, follow these steps:

1. Ensure you have Python 3.7 or later installed on your system.

2. Install jBark and its dependencies:

   ```
   pip install jbark numpy torch scipy librosa resampy
   ```

   Note: jBark uses CPU for computations by default. If you want to use GPU acceleration, make sure to install the appropriate CUDA-enabled version of PyTorch.

## Usage Instructions

Here's a basic example of how to use jBark:

```python
from jbark import JBark

# Initialize jBark
jbark = JBark()

# Generate audio from text
text = "Hello, this is a test of jBark text-to-speech."
output_path = "output.wav"
audio_array = jbark.generate_audio(text, output_path)

# Extract voice characteristics
sample_audio = "sample_voice.wav"
voice_chars = jbark.simple_voice_clone(sample_audio)

# Generate audio with simple voice conversion
converted_text = "This is speech using simple voice conversion."
converted_output_path = "converted_output.wav"
converted_audio = jbark.generate_with_cloned_voice(converted_text, voice_chars, converted_output_path)

# List supported languages
languages = jbark.list_supported_languages()
print("Supported languages:", languages)
```

This example demonstrates the basic workflow of generating speech, extracting voice characteristics, and applying simple voice conversion to the generated audio.

## Configuration and Customization

jBark provides several options for customization:

1. **Warning Suppression**: By default, jBark suppresses common warnings. This behavior is handled internally and doesn't require user configuration.

2. **Voice Presets**: When generating audio, you can specify a voice preset:

   ```python
   audio_array = jbark.generate_audio(text, history_prompt="v2/en_speaker_6")
   ```

3. **Voice Conversion Parameters**: You can adjust the strength of pitch shifting and tempo adjustment by modifying the `simple_voice_conversion` method in the `JBark` class.

## API Reference

### JBark Class

#### `__init__(self)`
Initializes the JBark instance, suppresses warnings, and preloads necessary models.

#### `generate_audio(self, text_prompt: str, output_path: str = None, history_prompt: str = None) -> numpy.ndarray`
Generates audio from the given text prompt.

- `text_prompt`: The text to convert to speech.
- `output_path`: Optional. Path to save the generated audio.
- `history_prompt`: Optional. Voice preset to use.

Returns: Numpy array containing the audio data.

#### `simple_voice_clone(self, audio_path: str) -> dict`
Extracts basic voice characteristics from an audio sample.

- `audio_path`: Path to the audio sample for voice characteristic extraction.

Returns: Dictionary containing basic voice characteristics (pitch and tempo).

#### `generate_with_cloned_voice(self, text_prompt: str, voice_characteristics: dict, output_path: str) -> numpy.ndarray`
Generates audio using simple voice conversion based on extracted voice characteristics.

- `text_prompt`: The text to convert to speech.
- `voice_characteristics`: Dictionary containing voice characteristics (pitch and tempo).
- `output_path`: Path to save the generated audio.

Returns: Numpy array containing the audio data.

#### `simple_voice_conversion(self, audio: numpy.ndarray, voice_characteristics: dict) -> numpy.ndarray`
Applies simple voice conversion to the input audio based on the given voice characteristics.

- `audio`: Input audio array.
- `voice_characteristics`: Dictionary containing voice characteristics (pitch and tempo).

Returns: Converted audio array.

#### `custom_time_stretch(self, audio: numpy.ndarray, rate: float) -> numpy.ndarray`
Custom time stretching function using resampling.

- `audio`: Input audio array.
- `rate`: Time stretch rate.

Returns: Time-stretched audio array.

#### `list_supported_languages(self) -> dict`
Returns a dictionary of supported languages.

## Code Architecture and Design

jBark is designed with modularity and extensibility in mind. The main components are:

1. **JBark Class**: The central interface for all functionality.
2. **Bark Model**: Handles text-to-speech generation.
3. **Simple Voice Conversion Module**: Manages basic voice characteristic extraction and application.

The library follows a facade pattern, where the JBark class provides a simplified interface to the underlying text-to-speech and voice conversion systems.

## Testing and Debugging

jBark comes with two test suites:

1. **Basic Test Suite** (`test_jbark.py`):
   To run the basic test suite:

   ```
   python test_jbark.py
   ```

   This will launch an interactive menu allowing you to test various features of the jBark library.

2. **Expanded Test Suite** (`test2.py`):
   To run the expanded test suite:

   ```
   python test2.py
   ```

   This suite provides more comprehensive testing, including variations in audio generation, voice cloning, voice conversion, language support, error handling, and performance testing.

For debugging, you can use Python's built-in `pdb` module or an IDE like PyCharm or VSCode.

## Common Issues and FAQs

Q: Why is the audio generation slow?
A: Audio generation speed depends on your hardware. jBark uses CPU for computations by default. For faster processing, consider using a machine with a more powerful CPU or implementing GPU support.

Q: How effective is the simple voice conversion?
A: The simple voice conversion feature in jBark provides basic pitch and tempo adjustments. While it can alter some voice characteristics, it does not provide the same level of voice cloning quality as more advanced methods. Results may vary depending on the input text and target voice characteristics.

Q: How can I improve the quality of voice conversion?
A: Use high-quality audio samples for voice characteristic extraction, ideally with clear speech and minimal background noise. You can also experiment with adjusting the pitch shifting and tempo adjustment parameters in the `simple_voice_conversion` method for better results.

Q: How do I use different language models?
A: jBark supports multiple languages. You can specify the desired language when generating audio by using the appropriate language code in the history prompt. For example:

```python
audio = jbark.generate_audio("Bonjour!", history_prompt="v2/fr_speaker_1")
```

For a list of supported languages and their codes, use the `list_supported_languages()` method.

For more questions and answers, visit our GitHub issues page or join our community forum.

![image](https://github.com/user-attachments/assets/c85b28e1-91af-43df-a444-9b68b12cf131)
