# 🎤 Speaker Detector Server

**Speaker Detector Server** is a FastAPI-based service that performs **speaker diarization** (detecting and separating different speakers in an audio file) using **Pyannote-Audio**.

This guide provides a **step-by-step setup** for installing, running, and developing the project.

---

## 🚀 Features

- 📂 **Upload an audio file** via FastAPI.
- 🎙️ **Detect and separate speakers** in the file.
- 🔗 **Uses Hugging Face models** for advanced speaker recognition.
- 🖥️ **FastAPI-based backend** for easy deployment.

---

## 🛠️ **1. Prerequisites**

Before you start, make sure you have:

- **Python 3.10 or later**
- **Ubuntu 24 LTS (or another Linux-based OS recommended)**
- **Hugging Face account** (to access the diarization model).
  > **in case you use the already generated Hugging Face token you do not need to do this step**

---

## 🔧 **2. Installation**

### 2.1 **Clone the Repository**

```bash
YET NOT ON GITHUB !!!!
```

### 2.2 **Create a Virtual Environment**

- run these commands in the root of the project

```bash
python3 -m venv pyannote-env
source pyannote-env/bin/activate
```

### 2.3 **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🛠️ **3. Configuration**

### 3.1 **Create a `.env` File** or rename `.env.example` to `.env`

update the `.env` file with your Hugging Face access token

```ini
HUGGING_FACE_ACCESS_TOKEN= REPLACE_ACCESS_TOKEN_HERE
```

---

## 🚀 **4. Running the Server**

With everything set up, start the FastAPI server:

```bash
uvicorn app.main:app --reload --port 8000
```

If successful, you should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## 📤 **5. Testing the API**

### **5.1 Upload an Audio File (Using `curl`)**

```bash
curl -X POST -F "file=@/path/to/your/audio.wav" http://127.0.0.1:8000/diarize
```

### **5.2 Expected JSON Response**

```json
{
  "segments": [
    { "speaker": "SPEAKER_00", "start": 0.2, "end": 2.5 },
    { "speaker": "SPEAKER_01", "start": 3.0, "end": 6.7 }
  ]
}
```

### **5.3 Restarting the Server After Changes**

The server automatically reloads with:

```bash
uvicorn app.main:app --reload
```

---

## 🏗️ **7. Folder Structure**

```
speaker-detector-server/
├── app/                # Main FastAPI application
│   ├── main.py         # FastAPI server logic
│   ├── requirements.txt # Dependencies list
├── pyannote-env/       # Virtual environment (excluded from Git)
├── .env                # Hugging Face token (excluded from Git)
├── .gitignore          # Files to ignore in Git
└── README.md           # This file
```
