# 🌐 Deepfake Detection Web App — ResNeXt-LSTM Interface

This folder (`/deepfake_console`) contains a simple **Streamlit-based web application** for testing the **ResNeXt-LSTM deepfake detection model** developed in our course project.

---

## 🧠 Overview

The app provides an **interactive interface** to:

* Upload a short video clip (`.mp4`, `.avi`, `.mov`)
* Automatically process frames and extract features
* Run inference using the pretrained **ResNeXt-LSTM** model
* Display predicted label (Real/Fake) and confidence score

---

## ⚙️ Setup Instructions

### 1. Install dependencies


```bash
cd ..
pip install -r requirements.txt
```

---

### 2. Check your model path

Make sure that the trained model weights are available in:

```
model/weights/best_resnext_lstm.pth
```

If not, download or place your `.pth` file there.

---

### 3. Launch the website locally

Run on bash:

```bash
cd model
streamlit run app.py
```

This will automatically start a **local development server** and open the web interface in your browser.

If it doesn’t open automatically, visit:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🖥️ App Functionality

Once launched:

1. You’ll see a “Deepfake Detection Demo” page.
2. Click **Upload a Video** and select any short clip.
3. The app will:

   * Extract frames from the uploaded video
   * Pass them through the **ResNeXt-LSTM** model
   * Display the prediction result with a confidence score

Example output:

```
Prediction: FAKE
Confidence: 0.91
```

---


## 🧩 Common Issues

| Problem                                      | Solution                                                            |
| -------------------------------------------- | ------------------------------------------------------------------- |
| *ModuleNotFoundError: No module named 'src'* | Run `streamlit run app.py` **from project root**, not inside `/src` |
| *Model not found error*                      | Ensure `best_resnext_lstm.pth` is in `models/`                      |
| *App not loading in browser*                 | Open `http://localhost:8501` manually                               |
| *CUDA errors*                                | Run on CPU by editing `predict_video.py` to use `device='cpu'`      |

---

## 📦 Optional Deployment

You can temporarily deploy the app online using:

```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

Then share your public URL (e.g., via Streamlit Cloud or ngrok).

Example with ngrok:

```bash
ngrok http 8501
```

This will generate a temporary public link to your local app.

---

