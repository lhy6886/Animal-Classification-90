🎉 Final step for a freshman!

Deploy your 90-class animal recognizer on Android!  

---

📦 Step 1 – Create the `.tflite` model  

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('animal90_savedmodel')  # folder with saved_model.pb
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('animal90.tflite', 'wb') as f:
    f.write(tflite_model)

print('✅ animal90.tflite created!')
```

---

🏷️ Step 2 – Create the label file  

```python
import os

root = r'D:\your\exact\path\to\data\animals90\train'  # change to your real path
labels = sorted(os.listdir(root))

with open('labels.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(labels))

print('✅ labels.txt created!')
```

---

✅ Quick checklist  
1. `animal90_savedmodel/` must contain `saved_model.pb` + variables folder.  
2. `root` in the label script must point to your training folder (e.g., `data/animals90/train`).  
3. Both files (`animal90_final.tflite`, `labels.txt`) go into your Android project’s `assets/` folder.



🎯 Next checklist – what you should do right now!

1️⃣ Install Android Studio

• 📥 Official & CN mirror: https://developer.android.google.cn/studio/

• Run the installer and let it pull the Android SDK automatically.

2️⃣ 🆕 Create a new “Empty Activity” project (Kotlin → Minimum SDK 21+).

3️⃣ 📂 Copy your two files into place  

   ```
   app/
   └── src/
       └── main/
           └── assets/
               ├── animal90_final.tflite
               └── labels.txt
   ```

If the 📁 `assets/` folder does not exist, create it manually.

4️⃣ 🧩 In Android Studio, add the TensorFlow Lite dependency (version 2.17.0 or newer) in `app/build.gradle.kts` under `dependencies` I’ve already placed the TensorFlow Lite dependency line you need below—just copy and paste!

5️⃣ 🔄 Sync & build the project once; Android Studio will package the `.tflite` and `labels.txt` into the APK.

6️⃣ 📱 Connect an Android phone (USB-debugging on) and hit ▶️ Run. The app will install and be ready for offline inference!

That’s all you need before you start writing the Kotlin UI and inference logic.




🎉 Porting a 90-class Animal Classifier to Android – Freshman Edition

> I’m a first-year CS student. With the step-by-step help of Kimi (Moonshot AI, Beijing), I completed my very first “model → Android” journey.

All code is open-source; senior devs are welcome to PR & roast!

---

📁 File 1️⃣  `MainActivity.kt`
Where to open:

Android Studio → Project pane → expand `app > java > com.example.animalsid` → open `MainActivity.kt`.

```kotlin
package com.example.animalsid

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var tflite: Interpreter
    private val labels by lazy { assets.open("labels.txt").bufferedReader().readLines() }
    private val PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    private val REQUEST_CODE = 1001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (hasPermissions()) {
            loadModelAndStartCamera()
        } else {
            ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_CODE)
        }
    }

    private fun hasPermissions(): Boolean =
        PERMISSIONS.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            loadModelAndStartCamera()
        } else {
            Toast.makeText(this, "需要相机权限", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun loadModelAndStartCamera() {
        // 使用 AssetFileDescriptor 和 FileChannel 流式读取模型文件
        val assetManager = assets
        val inputStream = assetManager.openFd("animal90.tflite").createInputStream()
        val fileChannel = inputStream.channel
        val modelByteBuffer = ByteBuffer.allocateDirect(fileChannel.size().toInt())
        modelByteBuffer.order(ByteOrder.nativeOrder())
        fileChannel.read(modelByteBuffer)
        fileChannel.close()
        inputStream.close()

        tflite = Interpreter(modelByteBuffer)
        startCamera()
    }

    private fun startCamera() {
        val previewView = findViewById<PreviewView>(R.id.previewView)
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(224, 224))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
            val bitmap = imageProxy.toBitmap()
            val label = classify(bitmap)
            findViewById<TextView>(R.id.result).text = label
            imageProxy.close()
        }

        ProcessCameraProvider.getInstance(this).get().apply {
            unbindAll()
            bindToLifecycle(
                this@MainActivity,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }
    }

    private fun classify(bitmap: Bitmap): String {
        val bmp = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputBuffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4)
            .order(ByteOrder.nativeOrder())
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val px = bmp.getPixel(x, y)
                inputBuffer.putFloat(((px shr 16) and 0xFF) / 255f)
                inputBuffer.putFloat(((px shr 8) and 0xFF) / 255f)
                inputBuffer.putFloat((px and 0xFF) / 255f)
            }
        }
        inputBuffer.rewind()

        val output = Array(1) { FloatArray(labels.size) }
        tflite.run(inputBuffer, output)
        val idxMax = output[0].indices.maxByOrNull { output[0][it] } ?: 0
        return "${labels[idxMax]} ${(output[0][idxMax] * 100).toInt()}"
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer[nv21, 0, ySize]
        vBuffer[nv21, ySize, vSize]
        uBuffer[nv21, ySize + vSize, uSize]
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val yuv = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
    }
}

```

---

📁 File 2️⃣  `build.gradle.kts` (Module: app)
Where to open:

Android Studio → Project pane → expand `app` → open `build.gradle.kts`.

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.animalsid"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.animalsid"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    // 禁止压缩 tflite 和 txt
    aaptOptions {
        noCompress += "tflite"
        noCompress += "txt"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.camera:camera-core:1.3.0")
    implementation("androidx.camera:camera-camera2:1.3.0")
    implementation("androidx.camera:camera-lifecycle:1.3.0")
    implementation("androidx.camera:camera-view:1.3.0")
    implementation("org.tensorflow:tensorflow-lite:2.15.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
}

```

---

📁 File 3️⃣  `AndroidManifest.xml`
Where to open:

Android Studio → Project pane → expand `app > manifests` → open `AndroidManifest.xml`.

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-permission android:name="android.permission.CAMERA" />


    <uses-feature
        android:name="android.hardware.camera.any"
        android:required="true" />

    <application
        android:allowBackup="true"
        android:label="AnimalsID"
        android:supportsRtl="true"
        android:theme="@style/Theme.AppCompat.Light.NoActionBar">

        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>

```

---

🚀 Full Journey Recap
1. Train the model → export SavedModel → convert to `animal90.tflite`.  
2. Generate `labels.txt` from the training folder.  
3. Place both files into `app/src/main/assets/`.  
4. Add dependencies, permissions, and manifest flags shown above.  
5. Sync, build, run on your Android phone—done!

Thanks again to Kimi for walking me through every line.

Looking forward to learning more from the community!