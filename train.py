"""
BSR Noise Classifier — YAMNet Fine-Tuning Pipeline
====================================================
Supports up to 5 noise classes (e.g. IP, Rear, Sunroof, Steering, IRVM).
Includes aggressive augmentation to handle small datasets (~100 samples).

Requirements:
    pip install tensorflow tensorflow-hub audiomentations librosa numpy scikit-learn matplotlib

Dataset folder structure expected:
    data/
        ip_noise/        ← .wav files
        rear_noise/
        sunroof_noise/
        steering_noise/
        other/           ← add/remove folders as needed

Usage:
    python train.py
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import audiomentations as A


# ─────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ─────────────────────────────────────────────
DATA_DIR        = "data"                  # root folder with one subfolder per class
OUTPUT_DIR      = "output"                # where model + logs are saved
SAMPLE_RATE     = 16000                   # YAMNet expects 16 kHz mono
CLIP_DURATION   = 3.0                     # seconds per sample (pad/trim to this)
AUGMENT_FACTOR  = 8                       # how many augmented copies per real sample
BATCH_SIZE      = 16
EPOCHS          = 40
LEARNING_RATE   = 1e-4
N_FOLDS         = 5                       # stratified k-fold
YAMNET_URL      = "https://tfhub.dev/google/yamnet/1"


# ─────────────────────────────────────────────
# 1. AUGMENTATION PIPELINE
# ─────────────────────────────────────────────
augment = A.Compose([
    A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.6),
    A.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    A.AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=0.5),
    A.AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.4),
    A.Shift(min_shift=-0.2, max_shift=0.2, p=0.4),
    A.RoomSimulator(p=0.3),  # simulate different car cabin acoustics
])


def load_audio(filepath):
    """Load and normalize a wav file to 16kHz mono."""
    waveform, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    target_len = int(CLIP_DURATION * SAMPLE_RATE)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)


# ─────────────────────────────────────────────
# 2. LOAD DATASET + AUGMENT
# ─────────────────────────────────────────────
def build_dataset():
    class_names = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    print(f"\nDetected {len(class_names)} classes: {class_names}\n")

    all_waveforms = []
    all_labels    = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        print(f"  {class_name}: {len(files)} original samples")

        for fname in files:
            waveform = load_audio(os.path.join(class_dir, fname))

            # original sample
            all_waveforms.append(waveform)
            all_labels.append(label_idx)

            # augmented copies
            for _ in range(AUGMENT_FACTOR):
                augmented = augment(samples=waveform, sample_rate=SAMPLE_RATE)
                all_waveforms.append(augmented.astype(np.float32))
                all_labels.append(label_idx)

        print(f"    → {len(files) * (1 + AUGMENT_FACTOR)} total after augmentation")

    return np.array(all_waveforms), np.array(all_labels), class_names


# ─────────────────────────────────────────────
# 3. YAMNET FEATURE EXTRACTOR
# ─────────────────────────────────────────────
def extract_yamnet_embeddings(waveforms, yamnet_model):
    """
    Run each waveform through YAMNet and extract the 1024-dim embedding.
    YAMNet returns embeddings per 0.48s frame; we take the mean across frames.
    """
    embeddings = []
    for i, waveform in enumerate(waveforms):
        if i % 100 == 0:
            print(f"  Extracting embeddings {i}/{len(waveforms)}...")
        _, emb, _ = yamnet_model(waveform)
        embeddings.append(tf.reduce_mean(emb, axis=0).numpy())
    return np.array(embeddings)


# ─────────────────────────────────────────────
# 4. CLASSIFIER HEAD (on top of YAMNet embedding)
# ─────────────────────────────────────────────
def build_classifier(num_classes, embedding_dim=1024):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name="bsr_noise_classifier")
    return model


# ─────────────────────────────────────────────
# 5. FULL END-TO-END MODEL (YAMNet + classifier)
# ─────────────────────────────────────────────
def build_full_model(num_classes, yamnet_model):
    """
    Wraps YAMNet + classifier in a single Keras model for easy inference + TFLite export.
    Input: raw waveform (float32 array of shape [N])
    Output: softmax probabilities over noise classes
    """
    waveform_input = tf.keras.Input(shape=(int(CLIP_DURATION * SAMPLE_RATE),),
                                    dtype=tf.float32, name="waveform")

    # YAMNet embedding (non-trainable)
    def yamnet_embedding(waveform):
        _, emb, _ = yamnet_model(waveform)
        return tf.reduce_mean(emb, axis=0, keepdims=True)

    embedding = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(
            lambda w: tf.reduce_mean(yamnet_model(w)[1], axis=0),
            x, fn_output_signature=tf.float32
        )
    )(waveform_input)

    # Classifier head
    x = tf.keras.layers.Dense(256, activation='relu')(embedding)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = tf.keras.Model(inputs=waveform_input, outputs=output, name="bsr_full_model")
    return model


# ─────────────────────────────────────────────
# 6. TRAINING WITH K-FOLD VALIDATION
# ─────────────────────────────────────────────
def train_with_kfold(embeddings, labels, class_names):
    num_classes = len(class_names)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_histories = []
    fold_reports   = []
    best_val_acc   = 0.0
    best_weights   = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*50}")

        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx],     labels[val_idx]

        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)

        # Compute class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
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
                monitor='val_accuracy', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
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

        # Per-fold classification report
        y_pred = np.argmax(model.predict(X_val), axis=1)
        report = classification_report(y_val, y_pred, target_names=class_names)
        print(f"\nFold {fold+1} classification report:\n{report}")
        fold_reports.append(report)

        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")

    print(f"\nBest validation accuracy across folds: {best_val_acc:.4f}")
    return best_weights, fold_histories, num_classes


# ─────────────────────────────────────────────
# 7. PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_training(fold_histories, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, h in enumerate(fold_histories):
        axes[0].plot(h['accuracy'],     label=f'Fold {i+1} train')
        axes[0].plot(h['val_accuracy'], label=f'Fold {i+1} val', linestyle='--')
        axes[1].plot(h['loss'],         label=f'Fold {i+1} train')
        axes[1].plot(h['val_loss'],     label=f'Fold {i+1} val', linestyle='--')

    axes[0].set_title('Accuracy per fold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(fontsize=7)
    axes[1].set_title('Loss per fold')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    print(f"\nTraining curves saved to {path}")


# ─────────────────────────────────────────────
# 8. SAVE FINAL MODEL
# ─────────────────────────────────────────────
def save_final_model(best_weights, num_classes, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)

    # Save classifier weights
    classifier = build_classifier(num_classes)
    classifier.set_weights(best_weights)
    classifier.save(os.path.join(output_dir, "classifier_head"))
    print(f"Classifier head saved to {output_dir}/classifier_head/")

    # Save class names for reference in Android app
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Class names saved to {output_dir}/class_names.txt")

    return classifier


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YAMNet from TF Hub (requires internet first time only)...")
    yamnet_model = hub.load(YAMNET_URL)

    print("\nLoading and augmenting dataset...")
    waveforms, labels, class_names = build_dataset()
    print(f"\nTotal samples after augmentation: {len(waveforms)}")
    print(f"Class distribution: { {c: int(np.sum(labels==i)) for i, c in enumerate(class_names)} }")

    print("\nExtracting YAMNet embeddings (this takes a few minutes)...")
    embeddings = extract_yamnet_embeddings(waveforms, yamnet_model)
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"),     labels)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nStarting k-fold training...")
    best_weights, fold_histories, num_classes = train_with_kfold(
        embeddings, labels, class_names
    )

    plot_training(fold_histories, OUTPUT_DIR)
    classifier = save_final_model(best_weights, num_classes, OUTPUT_DIR, class_names)

    print("\n✓ Training complete. Next step: run convert_to_tflite.py")
