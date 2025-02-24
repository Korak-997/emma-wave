# 📖 EmmaWave Developer Documentation

## 📌 Overview

**EmmaWave** is a FastAPI-based application that performs **Speaker Diarization**, meaning it can detect and separate different speakers in an audio file. It uses **Pyannote-Audio**, a Hugging Face-based deep learning model, to achieve accurate speaker detection.

### 🚀 Features

- **📂 Upload Audio Files:** Users can upload `.wav` files for analysis.
- **🎙️ Speaker Identification:** Detects and separates speakers in an audio file.
- **🔗 Hugging Face Model:** Utilizes `pyannote/speaker-diarization` for speaker recognition.
- **🖥️ API Endpoints:** Provides REST API for audio processing.
- **📤 Log Management:** Saves logs asynchronously for analysis.
- **🔥 GPU Acceleration:** Supports CUDA for faster processing if a GPU is available.

---

## 🛠️ Installation Guide

### **Step 1: Update & Install System Packages**

On a fresh **Ubuntu 24 LTS** installation, run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git ffmpeg
```

- `python3, pip3, venv`: Required for running the app.
- `git`: Required if cloning from a repository.
- `ffmpeg`: Used for audio processing.

### **Step 2: Clone the Project (If Hosted on GitHub)**

```bash
git clone https://github.com/your-repo/emmawave.git
cd emmawave
```

### **Step 3: Create a Virtual Environment**

```bash
python3 -m venv pyannote-env
source pyannote-env/bin/activate
```

### **Step 4: Install Python Dependencies**

```bash
pip install -r requirements.txt
```

This installs:

- **FastAPI & Uvicorn** (for running the server)
- **Pyannote-Audio** (for speaker diarization)
- **Torch & Torchaudio** (for deep learning inference)
- **Soundfile, FFmpeg-Python, and NumPy** (for audio processing)

### **Step 5: Configure Environment Variables**

Copy the example `.env` file and modify it:

```bash
cp .env_example .env
nano .env
```

Modify the following values:

```ini
HUGGING_FACE_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
SERVER_IP="0.0.0.0"
ENABLE_LOGGING=true
USE_GPU=true
```

Ensure you replace `YOUR_ACCESS_TOKEN` with a valid Hugging Face API token.

---

## 🚀 Running the Application

### **Step 1: Activate the Virtual Environment**

```bash
source pyannote-env/bin/activate
```

### **Step 2: Start the Server**

```bash
PYTHONPATH=$(pwd) uvicorn app.main:app --host 0.0.0.0 --port 7000 --reload
```

- `PYTHONPATH=$(pwd)`: Ensures all modules are correctly recognized.
- `--host 0.0.0.0` allows external access.
- `--port 7000` sets the server port.
- `--reload` enables automatic reloading during development.

### **Step 3: Test API Endpoints**

#### **1. Upload an Audio File**

```bash
curl -X POST -F "file=@/path/to/audio.wav" http://127.0.0.1:7000/diarize/
```

#### **2. Fetch Logs**

```bash
curl http://127.0.0.1:7000/logs/
```

---

## 🛠️ Debugging NVIDIA Driver & Secure Boot Issues

### ✅ **Check If GPU is Detected**

Run:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

- **Expected Output:** `True`
- **If False:** The GPU is not detected properly.

### ✅ **Check Installed NVIDIA Drivers**

Check available drivers:

```bash
ubuntu-drivers devices
```

This will list available NVIDIA drivers and recommend the best one.

To check currently installed drivers:

```bash
dpkg -l | grep -i nvidia
```

### ✅ **Install or Update NVIDIA Drivers**

To install the recommended driver (e.g., `nvidia-driver-550`):

```bash
sudo apt install nvidia-driver-550
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

### ✅ **Disable Secure Boot (Recommended Fix)**

Secure Boot can block the NVIDIA driver. To disable it:

1. **Reboot** and enter the BIOS/UEFI settings.
2. Locate **Secure Boot** (usually under Boot or Security settings).
3. **Disable it** and **Save & Exit**.
4. Check again with:
   ```bash
   mokutil --sb-state
   ```
   - Expected output: `SecureBoot disabled`

### ✅ **Manually Reinstall NVIDIA Driver (If Needed)**

```bash
sudo apt remove --purge '^nvidia-.*'
sudo apt autoremove
sudo apt clean
sudo apt install nvidia-driver-550
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

### ✅ **Verify PyTorch CUDA Support**

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

- Expected Output:
  ```
  True
  1
  NVIDIA GeForce RTX 3060 Ti
  ```
