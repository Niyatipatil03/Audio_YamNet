package com.bsr.noiseclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.bsr.noiseclassifier.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.text.SimpleDateFormat
import java.util.*

/**
 * BSR Noise Classifier — Main Activity
 *
 * Screen layout (top → bottom):
 *   • Header: app title + live status indicator
 *   • Current prediction card: large noise label + confidence %
 *   • All-classes confidence bars (one per class)
 *   • Scrollable detection log (recent high-confidence events)
 *   • START / STOP button
 *
 * Audio pipeline:
 *   AudioRecord (16 kHz, 16-bit PCM, mono)
 *       ↓  chunks of ~100 ms
 *   Circular float buffer (48 000 samples = 3 s)
 *       ↓  every 1 s (sliding window)
 *   TFLite classifier (AudioClassifier)
 *       ↓
 *   UI update on main thread
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val PERMISSION_REQ      = 101
        private const val STEP_SAMPLES        = AudioClassifier.SAMPLE_RATE   // 1 s slide
        private const val MAX_LOG_ENTRIES     = 25
        private const val CONFIDENCE_THRESHOLD = 0.55f  // only log detections above this
    }

    // View binding
    private lateinit var binding: ActivityMainBinding

    // Model
    private lateinit var classifier: AudioClassifier

    // Per-class UI rows (built dynamically after classifier loads)
    private data class ClassRow(
        val root: View,
        val progressBar: ProgressBar,
        val tvPct: TextView
    )
    private val classRows = mutableListOf<ClassRow>()

    // Recording state
    private var recordJob: Job? = null
    private var isRecording    = false

    // Circular audio buffer — holds exactly one 3-second window
    private val audioBuffer    = FloatArray(AudioClassifier.WINDOW_SAMPLES)
    private var writePos       = 0
    private var samplesBuffered = 0
    private var samplesToNextInference = AudioClassifier.WINDOW_SAMPLES  // wait for first full window

    private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    // ── Lifecycle ─────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = AudioClassifier(this)
        buildClassRows()

        binding.btnToggle.setOnClickListener { onToggleClicked() }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) stopRecording()
        classifier.close()
    }

    // ── Permission handling ────────────────────────────────────────────────

    override fun onRequestPermissionsResult(
        code: Int, perms: Array<out String>, results: IntArray
    ) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == PERMISSION_REQ &&
            results.isNotEmpty() && results[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startRecording()
        } else {
            Toast.makeText(this, "Microphone permission is required", Toast.LENGTH_LONG).show()
        }
    }

    private fun hasMicPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED

    // ── UI construction ───────────────────────────────────────────────────

    /** Inflate one confidence-bar row per noise class. */
    private fun buildClassRows() {
        binding.confidenceContainer.removeAllViews()
        classRows.clear()

        for (name in classifier.classNames) {
            val row  = LayoutInflater.from(this)
                .inflate(R.layout.item_confidence_row, binding.confidenceContainer, false)
            row.findViewById<TextView>(R.id.tvClassName).text = name

            val pb  = row.findViewById<ProgressBar>(R.id.progressBar)
            val pct = row.findViewById<TextView>(R.id.tvPct)
            classRows.add(ClassRow(row, pb, pct))
            binding.confidenceContainer.addView(row)
        }
    }

    // ── Start / Stop logic ────────────────────────────────────────────────

    private fun onToggleClicked() {
        if (isRecording) {
            stopRecording()
        } else {
            if (!hasMicPermission()) {
                ActivityCompat.requestPermissions(
                    this, arrayOf(Manifest.permission.RECORD_AUDIO), PERMISSION_REQ
                )
            } else {
                startRecording()
            }
        }
    }

    private fun startRecording() {
        isRecording = true
        audioBuffer.fill(0f)
        writePos = 0
        samplesBuffered = 0
        samplesToNextInference = AudioClassifier.WINDOW_SAMPLES

        binding.btnToggle.text = "STOP"
        binding.tvStatus.text  = "Listening..."
        binding.statusDot.setBackgroundResource(R.drawable.dot_green)

        recordJob = lifecycleScope.launch(Dispatchers.IO) { captureLoop() }
    }

    private fun stopRecording() {
        isRecording = false
        recordJob?.cancel()

        binding.btnToggle.text = "START"
        binding.tvStatus.text  = "Stopped"
        binding.statusDot.setBackgroundResource(R.drawable.dot_red)
    }

    // ── Audio capture + inference loop ────────────────────────────────────

    /**
     * Runs entirely on an IO thread.
     * Reads raw PCM chunks from AudioRecord, fills the circular buffer,
     * and triggers inference every [STEP_SAMPLES] new samples.
     */
    private suspend fun captureLoop() {
        val minBuf    = AudioRecord.getMinBufferSize(
            AudioClassifier.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        // Use ~100 ms chunks so inference latency feels responsive
        val chunkSize = maxOf(minBuf, AudioClassifier.SAMPLE_RATE / 10)
        val shortBuf  = ShortArray(chunkSize)

        val recorder  = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            AudioClassifier.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            chunkSize * 2          // double-buffer to avoid overrun
        )

        recorder.startRecording()

        try {
            while (isRecording && isActive) {
                val read = recorder.read(shortBuf, 0, chunkSize)
                if (read <= 0) continue

                // Convert 16-bit PCM to float32 in [-1, 1] and push to ring buffer
                for (i in 0 until read) {
                    audioBuffer[writePos] = shortBuf[i] / 32_768f
                    writePos = (writePos + 1) % AudioClassifier.WINDOW_SAMPLES
                }
                samplesBuffered          += read
                samplesToNextInference   -= read

                // Fire inference once we have accumulated STEP_SAMPLES new samples
                // and have at least one full 3-second window in the buffer
                if (samplesToNextInference <= 0 &&
                    samplesBuffered >= AudioClassifier.WINDOW_SAMPLES
                ) {
                    samplesToNextInference = STEP_SAMPLES
                    runInference()
                }
            }
        } finally {
            recorder.stop()
            recorder.release()
        }
    }

    private suspend fun runInference() {
        // Linearise the circular buffer into a contiguous snapshot
        val snapshot = FloatArray(AudioClassifier.WINDOW_SAMPLES)
        val tail     = AudioClassifier.WINDOW_SAMPLES - writePos
        System.arraycopy(audioBuffer, writePos, snapshot, 0, tail)
        System.arraycopy(audioBuffer, 0, snapshot, tail, writePos)

        val results = classifier.classify(snapshot)

        withContext(Dispatchers.Main) { updateUI(results) }
    }

    // ── UI update ─────────────────────────────────────────────────────────

    private fun updateUI(results: List<ClassificationResult>) {
        if (results.isEmpty()) return

        val top = results[0]

        // Large prediction display
        binding.tvTopLabel.text = top.label
        binding.tvTopConf.text  = "${(top.confidence * 100).toInt()}% confidence"

        // Highlight top prediction in accent colour; dim the rest
        val accentColor   = ContextCompat.getColor(this, R.color.accent_blue)
        val defaultColor  = ContextCompat.getColor(this, R.color.text_primary)

        val confByLabel = results.associate { it.label to it.confidence }

        for ((i, name) in classifier.classNames.withIndex()) {
            val conf = confByLabel[name] ?: 0f
            val row  = classRows[i]
            row.progressBar.progress = (conf * 100).toInt()
            row.tvPct.text           = "${(conf * 100).toInt()}%"

            // Bold + accent colour for the top prediction row
            val isTop = (name == top.label)
            row.root.alpha = if (isTop) 1f else 0.65f
        }

        // Log confident detections
        if (top.confidence >= CONFIDENCE_THRESHOLD) {
            addLogEntry(top)
        }
    }

    private fun addLogEntry(result: ClassificationResult) {
        val ts   = timeFormat.format(Date())
        val conf = "${(result.confidence * 100).toInt()}%"

        val tv = TextView(this).apply {
            text      = "[$ts]  ${result.label.padEnd(20)}  $conf"
            textSize  = 13f
            typeface  = android.graphics.Typeface.MONOSPACE
            setPadding(0, 6, 0, 6)
            setTextColor(ContextCompat.getColor(context, R.color.text_primary))
        }

        // Insert at top so newest entry is always visible without scrolling
        binding.logContainer.addView(tv, 0)

        // Trim log to MAX_LOG_ENTRIES
        while (binding.logContainer.childCount > MAX_LOG_ENTRIES) {
            binding.logContainer.removeViewAt(binding.logContainer.childCount - 1)
        }
    }
}
