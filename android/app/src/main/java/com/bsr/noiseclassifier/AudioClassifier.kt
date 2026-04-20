package com.bsr.noiseclassifier

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Wraps the TFLite BSR noise classifier model.
 *
 * The model was exported by convert_to_tflite.py and expects:
 *   Input  : float32 waveform, shape [WINDOW_SAMPLES] (48 000 values = 3 s @ 16 kHz)
 *   Output : float32 softmax probabilities, shape [num_classes]
 *
 * Workflow:
 *   1. Create an instance (loads model + class names from assets/)
 *   2. Call classify(floatArray) with exactly WINDOW_SAMPLES values
 *   3. Receive a list of ClassificationResult sorted by descending confidence
 *   4. Call close() when done
 */
data class ClassificationResult(
    val label: String,
    val confidence: Float
)

class AudioClassifier(context: Context) {

    companion object {
        const val MODEL_FILE       = "bsr_noise_classifier_quantized.tflite"
        const val CLASS_NAMES_FILE = "class_names.txt"
        const val SAMPLE_RATE      = 16_000
        const val WINDOW_SECONDS   = 3
        const val WINDOW_SAMPLES   = SAMPLE_RATE * WINDOW_SECONDS   // 48 000
    }

    val classNames: List<String>
    private val interpreter: Interpreter

    init {
        classNames  = loadClassNames(context)
        interpreter = buildInterpreter(context)
    }

    // ── private helpers ────────────────────────────────────────────────────

    private fun loadClassNames(context: Context): List<String> =
        context.assets.open(CLASS_NAMES_FILE)
            .bufferedReader()
            .readLines()
            .filter  { it.isNotBlank() }
            .map     { formatLabel(it.trim()) }

    /** "ip_noise" → "IP Noise", "sunroof_noise" → "Sunroof Noise", etc. */
    private fun formatLabel(raw: String): String =
        raw.replace('_', ' ')
            .split(' ')
            .joinToString(" ") { word ->
                // Keep common abbreviations all-caps
                when (word.uppercase()) {
                    "IP", "IRVM", "BSR" -> word.uppercase()
                    else -> word.replaceFirstChar { it.uppercase() }
                }
            }

    private fun buildInterpreter(context: Context): Interpreter {
        val afd    = context.assets.openFd(MODEL_FILE)
        val buffer = FileInputStream(afd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
        val options = Interpreter.Options().apply {
            numThreads = 2          // 2 threads is a good balance on mid-range phones
            useNNAPI   = false      // CPU-only — avoids NNAPI quirks on diverse hardware
        }
        return Interpreter(buffer, options)
    }

    // ── public API ─────────────────────────────────────────────────────────

    /**
     * Classify [waveform], which must contain exactly [WINDOW_SAMPLES] float32
     * samples normalised to the range [-1.0, 1.0].
     *
     * Returns results sorted by confidence (highest first).
     */
    fun classify(waveform: FloatArray): List<ClassificationResult> {
        require(waveform.size == WINDOW_SAMPLES) {
            "Expected $WINDOW_SAMPLES samples, got ${waveform.size}"
        }

        val inputBuf = ByteBuffer
            .allocateDirect(WINDOW_SAMPLES * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        waveform.forEach { inputBuf.putFloat(it) }
        inputBuf.rewind()

        val outputBuf = ByteBuffer
            .allocateDirect(classNames.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuf, outputBuf)
        outputBuf.rewind()

        val probs = FloatArray(classNames.size) { outputBuf.float }

        return classNames.mapIndexed { i, label ->
            ClassificationResult(label, probs[i])
        }.sortedByDescending { it.confidence }
    }

    fun close() = interpreter.close()
}
