import numpy as np
import sounddevice as sd

# Parameters
SAMPLE_RATE = 44100  # Hz
CHUNK = 1024         # Block size
CARRIER_FREQ = 17000 # Hz (mosquito tone)

def mosquito_modulation(indata, outdata, frames, time, status):
    global phase
    # indata: microphone input (shape: [frames, channels])
    # outdata: speaker output (must be filled)
    
    # Generate carrier signal with phase continuity
    t = (np.arange(frames) + phase) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * CARRIER_FREQ * t)
    phase += frames  # Update phase for next block
    
    # Amplitude modulation
    voice = indata[:, 0]
    voice_norm = (voice + 1) / 2  # Shift to [0, 1] range
    modulated = carrier * voice_norm
    
    # Send to output
    outdata[:, 0] = modulated.astype(np.float32)

# Initialize phase tracker
phase = 0

print("Real-time Mosquito Encoder - Press Ctrl+C to stop")
with sd.Stream(samplerate=SAMPLE_RATE,
               blocksize=CHUNK,
               dtype=np.float32,
               channels=1,
               callback=mosquito_modulation):
    sd.sleep(1000000)  # Run until interrupted
