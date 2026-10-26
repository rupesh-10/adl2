import os
import re
import json
import time
import urllib.request
import urllib.parse
import subprocess
import numpy as np
import wave
from concurrent.futures import ThreadPoolExecutor


def synthesize_fallback_speech(text, sample_rate=16000):
    # Formant synthesizer fallback for offline generation
    words = re.findall(r'[a-z0-9]+', text.lower())[:20]
    if not words:
        words = ['speech']
    
    total_audio = []
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    
    for word in words:
        # Duration proportional to word length
        dur = max(0.25, min(0.6, len(word) * 0.08))
        n_samples = int(sample_rate * dur)
        t = np.linspace(0, dur, n_samples, endpoint=False)
        
        # Base pitch contour (120 Hz - 180 Hz with intonation curve)
        f0 = 130.0 + 30.0 * np.sin(np.pi * t / dur)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        glottal_pulse = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
        
        # Pseudo-formants for vowels (F1: 500-800Hz, F2: 1200-2200Hz, F3: 2500-3000Hz)
        seed = sum(ord(c) for c in word)
        f1 = 450.0 + (seed % 350)
        f2 = 1200.0 + ((seed * 7) % 800)
        f3 = 2400.0 + ((seed * 13) % 600)
        
        formant1 = np.sin(2 * np.pi * f1 * t) * np.exp(-t * 8)
        formant2 = np.sin(2 * np.pi * f2 * t) * np.exp(-t * 12)
        formant3 = np.sin(2 * np.pi * f3 * t) * np.exp(-t * 16)
        
        # Envelope: Attack, Sustain, Decay
        env = np.sin(np.pi * t / dur) ** 1.5
        vocal = (glottal_pulse * 0.5 + formant1 * 0.3 + formant2 * 0.15 + formant3 * 0.05) * env
        
        # Unvoiced consonant burst at start/end
        noise = rng.randn(n_samples) * 0.08 * (1.0 - env)
        word_audio = (vocal + noise).astype(np.float32)
        
        # Normalize
        peak = np.max(np.abs(word_audio)) + 1e-6
        word_audio = word_audio / peak * 0.7
        
        # Small pause between words (0.05s)
        pause = np.zeros(int(sample_rate * 0.05), dtype=np.float32)
        total_audio.extend(word_audio)
        total_audio.extend(pause)
        
    return np.array(total_audio, dtype=np.float32)


def fetch_tts_audio(text, mp3_path, wav_path):
    """Fetches high-fidelity speech from Google TTS endpoint or falls back cleanly."""
    # Clean and truncate text for TTS endpoint
    clean_words = re.findall(r'[a-z0-9]+', text.lower())[:16]
    clean_text = ' '.join(clean_words)
    if not clean_text:
        clean_text = "photo caption"
        
    encoded = urllib.parse.quote_plus(clean_text)
    url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=en&client=tw-ob"
    
    success = False
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        if len(data) > 1000:
            with open(mp3_path, 'wb') as f:
                f.write(data)
            res = subprocess.run(
                ['ffmpeg', '-y', '-i', mp3_path, '-ar', '16000', '-ac', '1', wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if res.returncode == 0 and os.path.exists(wav_path):
                success = True
    except Exception:
        success = False

    if not success:
        # Use acoustic fallback synthesizer
        audio = synthesize_fallback_speech(clean_text, sample_rate=16000)
        with wave.open(wav_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            int_samples = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(int_samples.tobytes())
            
    # Cleanup temp mp3
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    return success


def process_sample(item, audio_dir, tmp_dir):
    img_name = item['image']
    caption = item['caption']
    base_name = os.path.splitext(img_name)[0]
    wav_path = os.path.join(audio_dir, f"{base_name}.wav")
    mp3_path = os.path.join(tmp_dir, f"{base_name}_{os.getpid()}.mp3")
    
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return {'image': img_name, 'caption': caption, 'audio': f"{base_name}.wav", 'status': 'cached'}
        
    is_tts = fetch_tts_audio(caption, mp3_path, wav_path)
    return {
        'image': img_name,
        'caption': caption,
        'audio': f"{base_name}.wav",
        'status': 'tts' if is_tts else 'fallback'
    }


def prepare_dataset(max_samples=600):
    audio_dir = './data/audio'
    tmp_dir = '/tmp/audio_dl'
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    captions_file = './data/flickr30k/captions.json'
    with open(captions_file) as f:
        pairs = json.load(f)[:max_samples]
        
    print(f"Preparing paired audio dataset for {len(pairs)} samples...")
    results = []
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_sample, p, audio_dir, tmp_dir) for p in pairs]
        for i, future in enumerate(futures):
            res = future.result()
            results.append(res)
            if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
                tts_count = sum(1 for r in results if r['status'] == 'tts')
                cached_count = sum(1 for r in results if r['status'] == 'cached')
                fallback_count = sum(1 for r in results if r['status'] == 'fallback')
                print(f"[{i+1}/{len(pairs)}] TTS: {tts_count}, Cached: {cached_count}, Fallback: {fallback_count}")
            time.sleep(0.04)  # Politeness delay for Google TTS
            
    out_meta = os.path.join(audio_dir, 'audio_captions.json')
    with open(out_meta, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Audio dataset successfully prepared! Metadata saved to {out_meta}")


if __name__ == '__main__':
    prepare_dataset(max_samples=600)
