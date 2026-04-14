"""
BSR Noise Classifier — TFLite Conversion
=========================================
Run this after train.py to produce a .tflite model ready for Android.

Usage:
    python convert_to_tflite.py
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


OUTPUT_DIR   = "output"
SAMPLE_RATE  = 16000
CLIP_DURATION = 3.0
YAMNET_URL   = "https://tfhub.dev/google/yamnet/1"


class BSRInferenceModel(tf.Module):
    """
    Wraps YAMNet + classifier into a single callable TF module.
    Input : float32 waveform of shape [48000] (3s at 16kHz)
    Output: softmax probabilities of shape [num_classes]
    """
    def __init__(self, yamnet_model, classifier):
        super().__init__()
        self.yamnet    = yamnet_model
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

    # Test the module before conversion
    dummy = tf.zeros([int(CLIP_DURATION * SAMPLE_RATE)], dtype=tf.float32)
    result = inference_model.infer(dummy)
    print(f"Test inference output shape: {result['predictions'].shape}")

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [inference_model.infer.get_concrete_function()],
        inference_model
    )

    if use_quantization:
        print("  Applying INT8 dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For full INT8 quantization (smaller + faster), provide representative data:
        embeddings = np.load(os.path.join(OUTPUT_DIR, "embeddings.npy"))

        def representative_dataset():
            for _ in range(100):
                idx = np.random.randint(len(embeddings))
                # Reconstruct a dummy waveform (real data would be better)
                yield [np.zeros(int(CLIP_DURATION * SAMPLE_RATE), dtype=np.float32)]

        # Dynamic range quantization (no representative data needed)
        # For full INT8, uncomment below:
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    suffix = "_quantized" if use_quantization else ""
    out_path = os.path.join(OUTPUT_DIR, f"bsr_noise_classifier{suffix}.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n✓ TFLite model saved: {out_path}")
    print(f"  Model size: {size_mb:.2f} MB")

    # Verify the TFLite model works
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=out_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Input  : name={input_details[0]['name']}, shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"  Output : name={output_details[0]['name']}, shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Run a test inference
    test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"  Test output (should sum ~1.0): {output}, sum={output.sum():.4f}")

    print("\n✓ Conversion complete. Copy the .tflite file to your Android project's assets/ folder.")
    print("  Also copy output/class_names.txt to Android assets/.")
    return out_path


if __name__ == "__main__":
    convert(use_quantization=True)
