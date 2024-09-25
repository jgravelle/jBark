# jbark.py - A Python wrapper for the Bark library.

import librosa
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io import wavfile
import warnings

class JBark:
    def __init__(self):
        print("Using CPU for computations.")
        self.SAMPLE_RATE = SAMPLE_RATE
        self._suppress_warnings()
        self.preload_models()

    def _suppress_warnings(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="You are using `torch.load`")
        warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
        warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

    def preload_models(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preload_models()

    def generate_audio(self, text_prompt, output_path=None, history_prompt=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_array = generate_audio(text_prompt, history_prompt=history_prompt)
        
        if output_path:
            wavfile.write(output_path, self.SAMPLE_RATE, audio_array)
        
        return audio_array

    def concatenate_sentence_segments(self, segments, pause_duration=0.1):
        """
        Concatenate multiple audio segments into a single sentence.
        
        :param segments: List of audio arrays to concatenate
        :param pause_duration: Duration of pause between segments in seconds
        :return: Concatenated audio array
        """
        pause_samples = int(pause_duration * SAMPLE_RATE)
        pause = np.zeros(pause_samples, dtype=np.int16)
        
        concatenated = np.concatenate([seg for pair in zip(segments, [pause] * len(segments)) for seg in pair])
        return concatenated[:-pause_samples]  # Remove the last pause

    def create_conversation(self, audio_segments, speaker_pause=0.5):
        """
        Create a conversation (e.g., podcast) from multiple audio segments.
        
        :param audio_segments: List of audio arrays representing each speaker's part
        :param speaker_pause: Duration of pause between speakers in seconds
        :return: Concatenated audio array of the full conversation
        """
        speaker_pause_samples = int(speaker_pause * SAMPLE_RATE)
        speaker_pause = np.zeros(speaker_pause_samples, dtype=np.int16)
        
        conversation = np.concatenate([seg for pair in zip(audio_segments, [speaker_pause] * len(audio_segments)) for seg in pair])
        return conversation[:-speaker_pause_samples]  # Remove the last pause

    def save_audio(self, audio_array, output_path):
        """
        Save an audio array to a WAV file.
        
        :param audio_array: The audio array to save
        :param output_path: The path where the WAV file should be saved
        """
        wavfile.write(output_path, SAMPLE_RATE, audio_array)

    def generate_long_audio(self, text, output_path=None, history_prompt=None, max_length=100):
        """
        Generate audio for long text by splitting it into smaller chunks.
        
        :param text: The long text to convert to speech
        :param output_path: Path to save the generated audio (optional)
        :param history_prompt: Voice preset to use (optional)
        :param max_length: Maximum length of each text chunk
        :return: Audio array of the full text
        """
        chunks = self._split_long_text(text, max_length)
        audio_segments = []
        
        for chunk in chunks:
            audio_segment = self.generate_audio(chunk, history_prompt=history_prompt)
            audio_segments.append(audio_segment)
        
        full_audio = self.concatenate_sentence_segments(audio_segments)
        
        if output_path:
            self.save_audio(full_audio, output_path)
        
        return full_audio

    def simple_voice_conversion(self, audio, characteristics):
        """
        Apply simple voice conversion based on pitch and tempo.
        
        :param audio: Input audio array
        :param characteristics: Dictionary containing voice characteristics (pitch and/or tempo)
        :return: Converted audio array
        """
        # Convert audio to floating point and normalize
        audio_float = librosa.util.normalize(audio.astype(np.float32))
        
        # Pitch shift
        if 'pitch' in characteristics:
            n_steps = characteristics['pitch'] * 12  # Convert octaves to semitones
            audio_converted = librosa.effects.pitch_shift(audio_float, sr=self.SAMPLE_RATE, n_steps=n_steps)
        else:
            audio_converted = audio_float
        
        # Tempo adjustment
        if 'tempo' in characteristics:
            rate = characteristics['tempo'] / 100.0  # Assuming 100 is the base tempo
            audio_converted = librosa.effects.time_stretch(audio_converted, rate=rate)
        
        # Ensure the output has the same data type as the input
        audio_converted = (audio_converted * 32767).astype(np.int16)
        
        return audio_converted
    
    def _split_long_text(self, text, max_length=100):
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            if len(' '.join(current_chunk + [word])) > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks