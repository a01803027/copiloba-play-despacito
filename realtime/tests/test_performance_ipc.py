"""Exercise instrument, transport, fixed-tempo stems, and the native bridge."""
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave


def main():
    engine, library = map(lambda value: str(Path(value).resolve()), sys.argv[1:3])
    with TemporaryDirectory() as directory:
        root = Path(directory)
        stems = [root / f'stem{i}.wav' for i in range(4)]
        for i, path in enumerate(stems):
            # Distinct stereo pulses, two beats at 120 BPM.
            pulse = struct.pack('<hh', 500 + 100 * i, 800 + 100 * i)
            with wave.open(str(path), 'wb') as stem:
                stem.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
                stem.writeframes(pulse * 1200 + bytes((48000 - 1200) * 4))
        output, report = root / 'render.wav', root / 'metrics.json'
        result = subprocess.run(
            [sys.executable, '-m', 'composer.performance', '--mode', 'mock',
             '--seconds', '3', '--offline-audio', '--record', str(output),
             '--metrics', str(report), '--engine', engine, '--library', library,
             '--stems', *map(str, stems)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        metrics = json.loads(report.read_text())
        assert all(count > 3 for count in metrics['hits']), metrics
        for key in ('drops', 'invalid', 'late', 'scheduler_overflows', 'stem_starved_frames'):
            assert metrics[key] == 0, metrics
        with wave.open(str(output), 'rb') as audio:
            assert audio.getframerate() == 48000 and audio.getnchannels() == 2
            assert audio.getnframes() >= 120000
            first = audio.readframes(1200)
            assert max(abs(s[0]) for s in struct.iter_unpack('<h', first)) > 100
            assert any(audio.readframes(48000))


if __name__ == '__main__':
    main()
