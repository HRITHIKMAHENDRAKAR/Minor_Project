@echo off
python -m pip install audio-separator[cpu] torch torchaudio soundfile numpy typeguard==4.3.0
python test_separation.py > output.log 2>&1
echo Done.
