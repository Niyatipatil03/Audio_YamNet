"""
BSR Noise Classifier — TFLite Conversion
=========================================
Run this after train.py to produce a .tflite model ready for Android.

Usage:
    python convert_to_tflite.py
"""

import os
import random
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub


OUTPUT_DIR    = "output"
DATA_DIR      = "data"          # same as in train.py — used for calibration samples
SAMPLE_RATE   = 16000
CLIP_DURATION = 3.0
YAMNET_URL    = "https://tfhub.dev/google/yamnet/1"


class BSRInferenceModel(tf.Module):
    """
    Wraps YAMNet + classifier into a single callable TF module.
    Input : float32 waveform of shape [48000] (3 s at 16 kHz)
    Output: softmax probabilities of shape [num_classes]
    """
    def __init__(self, yamnet_model, classifier):
        super().__init__()
        self.yamnet     = yamnet_model
        self.classifier = classifier

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[int(CLIP_DURATION * SAMPLE_RATE)],
                      dtype=tf.float32, name="waveform")
    ])
    def infer(self, waveform):
        _, embeddings, _ = self.yamnet(waveform)
        mean_embedding   = tf.reduce_mean(embeddings, axis=0, keepdims=True)
        predictions      = self.classifier(mean_embedding, training=False)
        return {"predictions": predictions[0]}


def _load_wav(path):
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target_len  = int(CLIP_DURATION * SAMPLE_RATE)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)


def _collect_wav_paths(data_dir, max_files=200):
    """Walk data_dir and collect up to max_files .wav paths."""
    wav_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    random.shuffle(wav_files)
    return wav_files[:max_files]


def convert(use_quantization=True):
    print("Loading YAMNet...")
    yamnet_model = hub.load(YAMNET_URL)

    print("Loading trained classifier...")
    classifier = tf.keras.models.load_model(
        os.path.join(OUTPUT_DIR, "classifier_head")
    )
    classifier.trainable = False

    print("Building inference module...")
    inference_model = BSRInferenceModel(yamnet_model, classifier)

    # Smoke-test the module before conversion
    dummy  = tf.zeros([int(CLIP_DURATION * SAMPLE_RATE)], dtype=tf.float32)
    result = inference_model.infer(dummy)
    print(f"Test inference output shape: {result['predictions'].shape}")

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [inference_model.infer.get_concrete_function()],
        inference_model
    )

    if use_quantization:
        print("  Applying dynamic-range INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # ── Representative dataset using REAL audio for proper calibration ──
        wav_paths = _collect_wav_paths(DATA_DIR)
        if wav_paths:
            print(f"  Calibrating on {len(wav_paths)} real audio files from '{DATA_DIR}'")

            def representative_dataset():
                for p in wav_paths:
                    try:
                        yield [_load_wav(p)]
                    except Exception:
                        pass  # skip corrupt files silently
        else:
            print(f"  WARNING: No .wav files found in '{DATA_DIR}'. "
                  "Using silence for calibration — quality will be reduced.")

            def representative_dataset():
                for _ in range(100):
                    yield [np.zeros(int(CLIP_DURATION * SAMPLE_RATE), dtype=np.float32)]

        # Uncomment the two lines below to enable full INT8 (weights + activations).
        # Requires a representative dataset (already wired above).
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    suffix   = "_quantized" if use_quantization else ""
    out_path = os.path.join(OUTPUT_DIR, f"bsr_noise_classifier{suffix}.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n✓ TFLite model saved: {out_path}")
    print(f"  Model size: {size_mb:.2f} MB")

    # Verify the TFLite model
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=out_path)
    interpreter.allocate_tensors()
    inp  = interpreter.get_input_details()
    out  = interpreter.get_output_details()
    print(f"  Input  : name={inp[0]['name']}, shape={inp[0]['shape']}, dtype={inp[0]['dtype']}")
    print(f"  Output : name={out[0]['name']}, shape={out[0]['shape']}, dtype={out[0]['dtype']}")

    test_input = np.zeros(inp[0]['shape'], dtype=np.float32)
    interpreter.set_tensor(inp[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(out[0]['index']).flatten()
    print(f"  Test output (should sum ~1.0): {np.round(output, 3)}, sum={output.sum():.4f}")

    print("\n✓ Conversion complete.")
    print("  Copy the following files to your Android project's assets/ folder:")
    print(f"    {out_path}")
    print(f"    {os.path.join(OUTPUT_DIR, 'class_names.txt')}")
    return out_path


if __name__ == "__main__":
    convert(use_quantization=True)
