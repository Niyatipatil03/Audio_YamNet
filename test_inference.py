"""
BSR Noise Classifier — Quick Inference Test
============================================
Test your trained model on a single audio file before deploying to Android.

Usage:
    python test_inference.py path/to/test_audio.wav
"""

import sys
import os
import numpy as np
import librosa
import tensorflow as tf


OUTPUT_DIR       = "output"
SAMPLE_RATE      = 16000
CLIP_DURATION    = 3.0
TFLITE_PATH      = os.path.join(OUTPUT_DIR, "bsr_noise_classifier_quantized.tflite")
CLASS_NAMES_PATH = os.path.join(OUTPUT_DIR, "class_names.txt")


def load_audio(filepath):
    waveform, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    target_len = int(CLIP_DURATION * SAMPLE_RATE)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)


def predict(audio_path):
    # Load class names
    with open(CLASS_NAMES_PATH) as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load audio
    waveform = load_audio(audio_path)
    print(f"Audio loaded: {len(waveform)} samples @ {SAMPLE_RATE} Hz")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected_shape = tuple(input_details[0]['shape'])
    print(f"Model expects input shape: {expected_shape}")

    # Reshape waveform to match the model's expected input shape exactly.
    # BSRInferenceModel.infer was traced with shape=[WINDOW_SAMPLES] (1-D),
    # so do NOT add a batch dimension here.
    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])

    # probs may be shape [num_classes] or [1, num_classes] depending on TFLite version
    probs = probs.flatten()

    # Print results
    print("\n── BSR Noise Prediction ──────────────────")
    for name, prob in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:<20} {bar:<30} {prob*100:5.1f}%")

    top_class = class_names[np.argmax(probs)]
    top_conf  = np.max(probs)
    print(f"\n  → Predicted: {top_class} ({top_conf*100:.1f}% confidence)")
    print("──────────────────────────────────────────")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <path_to_wav>")
        sys.exit(1)
    predict(sys.argv[1])
