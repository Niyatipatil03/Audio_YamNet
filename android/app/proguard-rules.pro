# Keep TensorFlow Lite classes so the interpreter works after minification
-keep class org.tensorflow.** { *; }
-keep interface org.tensorflow.** { *; }
-dontwarn org.tensorflow.**
