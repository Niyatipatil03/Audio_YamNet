"""
BSR Noise Classifier — YAMNet Fine-Tuning Pipeline
====================================================
Supports any number of noise classes (e.g. IP, Rear, Sunroof, Steering, IRVM, …).
Uses aggressive augmentation to compensate for small datasets (~100 samples).

WHY YAMNET?
-----------
YAMNet was pre-trained on AudioSet (2 M clips, 521 classes) and produces a
1024-dimensional embedding that encodes rich spectral + temporal audio features.
Training a plain CNN or SVM on raw spectrograms from 100 samples cannot
compete with this — the embeddings already "understand" audio far better than
anything you can learn from scratch on small data.

POOLING MODES
-------------
POOLING_MODE = 'mean'   — average all frame embeddings (default, fast)
POOLING_MODE = 'max'    — per-dimension maximum (often better for transient noises)
POOLING_MODE = 'concat' — [mean ‖ max], doubles embedding dim to 2048

Dataset folder structure expected:
    data/
        ip_noise/        ← .wav files
        rear_noise/
        sunroof_noise/
        steering_noise/
        irvm_noise/
        other/           ← add/remove folders as needed

Requirements:
    pip install -r requirements.txt

Usage:
    python train.py
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import audiomentations as A


# ─────────────────────────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ─────────────────────────────────────────────────────────────────
DATA_DIR       = "data"       # root folder with one subfolder per class
OUTPUT_DIR     = "output"     # where model + logs are saved
SAMPLE_RATE    = 16000        # YAMNet expects 16 kHz mono
CLIP_DURATION  = 3.0          # seconds per sample (pad / trim to this length)
AUGMENT_FACTOR = 10           # augmented copies per real sample
                               # 100 samples × 10 → 1100 total (still small, but workable)
BATCH_SIZE     = 16
EPOCHS         = 60
LEARNING_RATE  = 1e-4
N_FOLDS        = 5            # stratified k-fold cross-validation

# How to pool YAMNet's per-frame embeddings into a fixed-size vector.
# 'mean'   → 1024-dim  (good for continuous hum / drone noises)
# 'max'    → 1024-dim  (good for short clicks, rattles, squeaks)
# 'concat' → 2048-dim  (combines both, usually best — try this if mean/max under-perform)
POOLING_MODE = 'concat'

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"


# ─────────────────────────────────────────────────────────────────
# 1. AUGMENTATION PIPELINE
#    Simulate the variety of real BSR-track conditions:
#      • vehicle speed changes  → time-stretch / pitch-shift
#      • road + wind noise      → gaussian noise / SNR
#      • cabin reverb variation → RoomSimulator (requires pyroomacoustics)
#      • recording position     → time-shift
# ─────────────────────────────────────────────────────────────────
augment = A.Compose([
    A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.6),
    A.PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.5),
    A.AddGaussianSNR(min_snr_db=8, max_snr_db=35, p=0.4),
    A.Shift(min_shift=-0.25, max_shift=0.25, p=0.5),
    A.RoomSimulator(p=0.3),   # simulates different car-cabin acoustic environments
                               # needs pyroomacoustics — see requirements.txt
])


def load_audio(filepath):
    """Load and normalise a wav file to 16 kHz mono float32."""
    waveform, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    target_len  = int(CLIP_DURATION * SAMPLE_RATE)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 2. LOAD DATASET + AUGMENT
# ─────────────────────────────────────────────────────────────────
def build_dataset():
    class_names = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    print(f"\nDetected {len(class_names)} classes: {class_names}\n")

    all_waveforms, all_labels = [], []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".wav", ".mp3", ".flac"))]
        print(f"  {class_name}: {len(files)} original samples")

        for fname in files:
            waveform = load_audio(os.path.join(class_dir, fname))
            all_waveforms.append(waveform)
            all_labels.append(label_idx)

            for _ in range(AUGMENT_FACTOR):
                aug = augment(samples=waveform, sample_rate=SAMPLE_RATE)
                all_waveforms.append(aug.astype(np.float32))
                all_labels.append(label_idx)

        print(f"    → {len(files) * (1 + AUGMENT_FACTOR)} total after augmentation")

    return np.array(all_waveforms), np.array(all_labels), class_names


# ─────────────────────────────────────────────────────────────────
# 3. YAMNET EMBEDDING EXTRACTION
#    POOLING_MODE controls how per-frame embeddings are collapsed
#    to a single fixed-size vector.
# ─────────────────────────────────────────────────────────────────
def _pool(emb_seq):
    """emb_seq: float32 [num_frames, 1024] → 1D vector."""
    mean_vec = tf.reduce_mean(emb_seq, axis=0).numpy()
    max_vec  = tf.reduce_max(emb_seq,  axis=0).numpy()

    if POOLING_MODE == 'mean':
        return mean_vec
    elif POOLING_MODE == 'max':
        return max_vec
    else:  # 'concat'
        return np.concatenate([mean_vec, max_vec])


def extract_yamnet_embeddings(waveforms, yamnet_model):
    embeddings = []
    for i, waveform in enumerate(waveforms):
        if i % 100 == 0:
            print(f"  Extracting embeddings {i}/{len(waveforms)}...")
        _, emb, _ = yamnet_model(waveform)
        embeddings.append(_pool(emb))
    return np.array(embeddings)


# ─────────────────────────────────────────────────────────────────
# 4. CLASSIFIER HEAD
# ─────────────────────────────────────────────────────────────────
def build_classifier(num_classes):
    embedding_dim = 1024 if POOLING_MODE in ('mean', 'max') else 2048

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name="bsr_noise_classifier")
    return model


# ─────────────────────────────────────────────────────────────────
# 5. K-FOLD TRAINING
# ─────────────────────────────────────────────────────────────────
def train_with_kfold(embeddings, labels, class_names):
    num_classes = len(class_names)
    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_histories, fold_reports = [], []
    best_val_acc = 0.0
    best_weights = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*50}")

        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx],     labels[val_idx]

        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)

        class_weights_arr = compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(enumerate(class_weights_arr))

        model = build_classifier(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=12, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1
            ),
        ]

        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        fold_histories.append(history.history)

        y_pred  = np.argmax(model.predict(X_val), axis=1)
        report  = classification_report(y_val, y_pred, target_names=class_names)
        print(f"\nFold {fold+1} classification report:\n{report}")
        fold_reports.append(report)

        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            print(f"  ✓ New best model (val_acc={val_acc:.4f})")

    print(f"\nBest validation accuracy across all folds: {best_val_acc:.4f}")
    return best_weights, fold_histories, num_classes


# ─────────────────────────────────────────────────────────────────
# 6. PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────────────
def plot_training(fold_histories, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, h in enumerate(fold_histories):
        axes[0].plot(h['accuracy'],     label=f'Fold {i+1} train')
        axes[0].plot(h['val_accuracy'], label=f'Fold {i+1} val', linestyle='--')
        axes[1].plot(h['loss'],         label=f'Fold {i+1} train')
        axes[1].plot(h['val_loss'],     label=f'Fold {i+1} val', linestyle='--')

    for ax, title in zip(axes, ['Accuracy per fold', 'Loss per fold']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    print(f"\nTraining curves saved to {path}")


# ─────────────────────────────────────────────────────────────────
# 7. SAVE FINAL MODEL
# ─────────────────────────────────────────────────────────────────
def save_final_model(best_weights, num_classes, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)

    classifier = build_classifier(num_classes)
    classifier.set_weights(best_weights)
    classifier.save(os.path.join(output_dir, "classifier_head"))
    print(f"Classifier head saved → {output_dir}/classifier_head/")

    # Save class names and the pooling mode for the Android app
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    with open(os.path.join(output_dir, "pooling_mode.txt"), "w") as f:
        f.write(POOLING_MODE + "\n")

    print(f"Class names saved → {output_dir}/class_names.txt")
    print(f"Pooling mode saved → {output_dir}/pooling_mode.txt  ({POOLING_MODE})")
    return classifier


# ─────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YAMNet from TF Hub (downloads once, then cached)...")
    yamnet_model = hub.load(YAMNET_URL)

    print(f"\nLoading + augmenting dataset  (AUGMENT_FACTOR={AUGMENT_FACTOR})...")
    waveforms, labels, class_names = build_dataset()
    print(f"\nTotal samples after augmentation : {len(waveforms)}")
    print(f"Class distribution               : {dict(zip(class_names, np.bincount(labels)))}")

    print(f"\nExtracting YAMNet embeddings  (pooling={POOLING_MODE})...")
    embeddings = extract_yamnet_embeddings(waveforms, yamnet_model)
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"),     labels)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nStarting k-fold training...")
    best_weights, fold_histories, num_classes = train_with_kfold(
        embeddings, labels, class_names
    )

    plot_training(fold_histories, OUTPUT_DIR)
    save_final_model(best_weights, num_classes, OUTPUT_DIR, class_names)

    print("\n✓ Training complete.  Next step: run  python convert_to_tflite.py")
