# jBark Library Documentation
![image](https://github.com/user-attachments/assets/9a021ada-b9d5-4e78-98cb-ec8695a0a189)

## Table of Contents
1. [Overview and Introduction](#overview-and-introduction)
2. [Installation Guide](#installation-guide)
3. [Usage Instructions](#usage-instructions)
4. [Configuration and Customization](#configuration-and-customization)
5. [API Reference](#api-reference)
6. [Code Architecture and Design](#code-architecture-and-design)
7. [Contribution Guide](#contribution-guide)
8. [Testing and Debugging](#testing-and-debugging)
9. [Licensing and Legal Considerations](#licensing-and-legal-considerations)
10. [Versioning and Updates](#versioning-and-updates)
11. [Common Issues and FAQs](#common-issues-and-faqs)

## Overview and Introduction

jBark is a powerful Python library that combines the capabilities of the original Bark text-to-speech project with simple voice conversion features. It provides a seamless interface for generating high-quality speech from text, extracting basic voice characteristics, and applying these characteristics to generated audio.

Key features of jBark include:
- Text-to-speech generation using the Bark model
- Simple voice characteristic extraction
- Basic voice conversion using pitch shifting and tempo adjustment
- Support for multiple languages

Whether you're developing a virtual assistant, creating audiobooks, or working on any project that requires flexible and high-quality speech synthesis, jBark provides the tools you need to bring your ideas to life.

## Installation Guide

To install jBark, follow these steps:

1. Ensure you have Python 3.7 or later installed on your system.

2. Install jBark using pip:

   ```
   pip install jbark
   ```

   If you want to install the latest development version from GitHub:

   ```
   pip install git+https://github.com/yourusername/jbark.git
   ```

3. Install additional dependencies:

   ```
   pip install torch torchaudio librosa
   ```

   Note: For GPU acceleration, make sure to install the appropriate CUDA version compatible with your PyTorch installation.

### Troubleshooting

- If you encounter issues with PyTorch installation, refer to the official PyTorch documentation for your specific operating system and CUDA version.
- For librosa-related issues, ensure you have the latest version installed and check their documentation for any known issues with your system configuration.

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

1. **Model Selection**: You can choose between different Bark model sizes by setting the `SUNO_USE_SMALL_MODELS` environment variable:

   ```python
   import os
   os.environ["SUNO_USE_SMALL_MODELS"] = "True"  # Use smaller models for faster inference
   ```

2. **Voice Presets**: When generating audio, you can specify a voice preset:

   ```python
   audio_array = jbark.generate_audio(text, history_prompt="en_speaker_6")
   ```

3. **Voice Conversion Parameters**: You can adjust the strength of pitch shifting and tempo adjustment in the `simple_voice_conversion` method by modifying the code directly.

## API Reference

### JBark Class

#### `__init__(self)`
Initializes the JBark instance and preloads necessary models.

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

#### `list_supported_languages(self) -> dict`
Returns a dictionary of supported languages.

## Code Architecture and Design

jBark is designed with modularity and extensibility in mind. The main components are:

1. **JBark Class**: The central interface for all functionality.
2. **Bark Model**: Handles text-to-speech generation.
3. **Simple Voice Conversion Module**: Manages basic voice characteristic extraction and application.

The library follows a facade pattern, where the JBark class provides a simplified interface to the underlying text-to-speech and voice conversion systems.

## Contribution Guide

We welcome contributions to jBark! Here's how you can contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Write your code, following the project's coding standards (PEP 8).
4. Add or update tests as necessary.
5. Update the documentation to reflect your changes.
6. Submit a pull request with a clear description of your changes.

For more details, see the CONTRIBUTING.md file in the project repository.

## Testing and Debugging

To run the test suite:

```
python test_jbark.py
```

This will launch an interactive menu allowing you to test various features of the jBark library.

For debugging, you can use Python's built-in `pdb` module or an IDE like PyCharm or VSCode. Set the `JBARK_DEBUG` environment variable to enable verbose logging:

```
export JBARK_DEBUG=1
```

## Licensing and Legal Considerations

jBark is licensed under the MIT License. See the LICENSE file for full details.

Note that jBark uses the following third-party libraries, each with its own license:
- Bark: MIT License
- PyTorch: BSD License
- librosa: ISC License

Ensure compliance with these licenses when using jBark in your projects.

## Versioning and Updates

jBark follows semantic versioning (MAJOR.MINOR.PATCH). See the CHANGELOG.md file for a detailed history of changes.

## Common Issues and FAQs

Q: Why is the audio generation slow?
A: Audio generation speed depends on your hardware. Using a GPU can significantly speed up the process. Also, consider using the smaller models by setting `SUNO_USE_SMALL_MODELS=True`.

Q: Can I use jBark for commercial projects?
A: Yes, jBark is licensed under the MIT License, which allows commercial use. However, ensure you comply with the licenses of all dependencies as well.

Q: How effective is the simple voice conversion?
A: The simple voice conversion feature in jBark provides basic pitch and tempo adjustments. While it can alter some voice characteristics, it does not provide the same level of voice cloning quality as more advanced methods. Results may vary depending on the input text and target voice characteristics.

Q: How can I improve the quality of voice conversion?
A: Use high-quality audio samples for voice characteristic extraction, ideally with clear speech and minimal background noise. You can also experiment with adjusting the pitch shifting and tempo adjustment parameters in the `simple_voice_conversion` method for better results.

For more questions and answers, visit our GitHub issues page or join our community forum.

![image](https://github.com/user-attachments/assets/5387b408-bf11-4458-9f1f-94cac2e79dd0)
