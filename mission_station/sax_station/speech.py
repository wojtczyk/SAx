from __future__ import annotations

from pathlib import Path

import sounddevice as sd
import soundfile as sf


class TranscriptionUnavailable(RuntimeError):
    pass


class SpeechRecorder:
    def __init__(self, data_dir: Path, sample_rate: int = 16_000) -> None:
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.audio_dir = data_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

    def record(self, seconds: int) -> Path:
        frames = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        audio_path = self.audio_dir / "last_voice_note.wav"
        sf.write(audio_path, frames, self.sample_rate)
        return audio_path

    def transcribe(self, audio_path: Path) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionUnavailable(
                "Speech transcription needs faster-whisper. Install requirements, then try again."
            ) from exc

        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(audio_path), beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        if not transcript:
            raise TranscriptionUnavailable("No speech was detected in the recording.")
        return transcript
