import importlib.util
from pathlib import Path
import numpy as np


audio_mixer_spec = importlib.util.spec_from_file_location(
    "conversation_recorder_audio_mixer",
    Path(__file__).parent.parent / "audio_mixer.py",
)
audio_mixer_module = importlib.util.module_from_spec(audio_mixer_spec)
audio_mixer_spec.loader.exec_module(audio_mixer_module)
AudioMixer = audio_mixer_module.AudioMixer


def pcm(samples):
    return np.array(samples, dtype=np.int16).tobytes()


def samples(audio):
    return np.frombuffer(audio, dtype=np.int16)


def test_mixer_mixes_two_sources():
    mixer = AudioMixer(sample_rate=1000)

    mixer.push_audio("user", pcm([100] * 50), 1000)
    mixer.push_audio("0", pcm([200] * 50), 1000)

    mixed = samples(mixer.mix_samples(50))
    assert len(mixed) == 50
    assert np.all(mixed == 300)


def test_mixer_pads_silence_when_source_exhausted():
    mixer = AudioMixer(sample_rate=1000)

    mixer.push_audio("user", pcm([100] * 20), 1000)
    mixer.push_audio("0", pcm([200] * 40), 1000)

    mixed = samples(mixer.mix_samples(40))
    assert len(mixed) == 40
    # First 20 samples: 100 + 200 = 300
    assert np.all(mixed[:20] == 300)
    # Next 20 samples: 0 + 200 = 200
    assert np.all(mixed[20:] == 200)


def test_mixer_returns_silence_when_empty():
    mixer = AudioMixer(sample_rate=1000)

    mixed = samples(mixer.mix_samples(40))
    assert len(mixed) == 40
    assert np.all(mixed == 0)


def test_mixer_flush_source():
    mixer = AudioMixer(sample_rate=1000)

    mixer.push_audio("0", pcm([200] * 100), 1000)
    mixer.flush_source("0")

    mixed = samples(mixer.mix_samples(50))
    assert np.all(mixed == 0)


def test_conversation_turn_taking_no_overlap():
    # 1000 Hz sample rate: 1000 samples = 1 sec
    mixer = AudioMixer(sample_rate=1000)

    # 1. User speaks for 1 sec (1000 samples with value 100)
    mixer.push_audio("user", pcm([100] * 1000), 1000)
    # Drain 1 sec
    chunk1 = samples(mixer.mix_samples(1000))
    assert np.all(chunk1 == 100)

    # 2. Silence for 1 sec (LLM thinking time)
    chunk2 = samples(mixer.mix_samples(1000))
    assert np.all(chunk2 == 0)

    # 3. AI responds for 1 sec (1000 samples with value 200)
    mixer.push_audio("0", pcm([200] * 1000), 1000)
    chunk3 = samples(mixer.mix_samples(1000))
    assert np.all(chunk3 == 200)


if __name__ == "__main__":
    test_mixer_mixes_two_sources()
    test_mixer_pads_silence_when_source_exhausted()
    test_mixer_returns_silence_when_empty()
    test_mixer_flush_source()
    test_conversation_turn_taking_no_overlap()
    print("All AudioMixer tests passed successfully!")
