# Real-Time American Sign Language (ASL) Recognition System

A real-time American Sign Language (ASL) fingerspelling recognition system that operates on standard 720p webcam input without dedicated GPU hardware. The pipeline integrates Google MediaPipe Hands for skeletal tracking, an invariant geometric classifier, temporal debounce buffering, and Google Gemini 3.1 Flash Lite for natural sentence synthesis with Text-to-Speech (TTS) playback.

---

## Table of Contents

1. Project Overview
2. System Architecture
3. Supported Vocabulary
4. System Requirements
5. Installation and Setup Guide (Step-by-Step)
6. How to Run the Application
7. Controls and Usage Guide
8. Gemini API Configuration (Optional)
9. Troubleshooting and Common Errors
10. Project Information

---

## 1. Project Overview

Direct communication between the Deaf and Hard-of-Hearing (DHH) community and the hearing public is frequently limited due to communication barriers. While standard deep learning video models demand high computational resources and dedicated graphics cards, this project implements a lightweight geometric pipeline capable of running smoothly on consumer laptop processors (CPUs).

Key achievements:

- Processes 720p webcam video at an average throughput of 37.0 FPS (range: 35 to 55 FPS) on CPU hardware.
- Recognizes 21 static alphabet letters and a 5.0-second open-palm space delimiter.
- Attains an overall empirical classification precision of 89.0%.
- Automatically reconstructs isolated letters into natural sentences in Vietnamese and English using Gemini Flash Lite, supplemented by an offline fallback dictionary.

---

## 2. System Architecture

The dataflow progresses sequentially through the following modules:

```text
[Webcam Video Input (720p @ 30-60 FPS)]
│
▼
[MediaPipe Hands Tracking] (Extracts 21 2D Landmark Coordinates)
│
▼
[Invariant Geometric Normalization] (Wrist Origin Shift & Metacarpal Scale Divisor)
│
▼
[Deterministic Rule Classifier] (Evaluates 21 Static Signs + Space Trigger)
│
▼
[Debounce Buffer State Machine] (12 Consecutive Frames >= 80% + 1.0s Cooldown)
│
▼
[Gemini 3.1 Flash Lite API] (Contextual Sentence Synthesis / Offline Dictionary)
│
▼
[Audio Output (TTS) & Web Interface] (Non-blocking Speech & HUD Display)
```

---

## 3. Supported Vocabulary

The system recognizes 21 static ASL alphabet signs and 1 continuous word delimiter:

| Sign | Description of Hand Posture |
| :--- | :--- |
| **A** | Tight fist with thumb resting straight up alongside the outer edge of the index finger. |
| **B** | Flat hand, all 4 fingers extended straight up touching, thumb folded across the palm. |
| **C** | All fingers and thumb curved into an open C-arc shape. |
| **D** | Index finger pointing straight up, remaining 3 fingers curled down, thumb touching middle finger. |
| **F** | Middle, ring, and pinky extended upright; tips of thumb and index finger touch (OK sign). |
| **G** | Three fingers folded down; index finger and thumb extended horizontally sideways. |
| **H** | Index and middle fingers extended together pointing horizontally sideways. |
| **I** | Only pinky finger extended straight up; remaining fingers curled into a fist. |
| **K** | Index and middle fingers extended upright in a V shape; thumb tip placed upright between them. |
| **L** | Thumb and index finger form a perpendicular right angle (90-degree L-shape). |
| **M** | Fist with 4 fingers folded; thumb extended and tucked between ring and pinky fingers. |
| **N** | Fist with 4 fingers folded; thumb extended and tucked between middle and ring fingers. |
| **O** | Fingertips curve down to meet the thumb tip, forming a closed circular ring. |
| **R** | Index and middle fingers extended upright with middle finger crossed behind index finger. |
| **S** | Fist with all 4 fingers curled down and thumb folded horizontally across the front of fingers. |
| **T** | Fist with 4 fingers folded; thumb extended and tucked between index and middle fingers. |
| **U** | Index and middle fingers extended straight up touching each other in parallel. |
| **V** | Victory sign; index and middle fingers pointing straight up spread apart in an open V. |
| **W** | Index, middle, and ring fingers extended upright in a W formation; pinky curled down. |
| **X** | Fist with lower half of index finger upright and upper half bent into a downward hook. |
| **Y** | Shaka sign; thumb and pinky extended outwards, middle three fingers folded down. |
| **SPACE** | Open hand with all 5 fingers extended steadily for 5.0 seconds to insert a word separator. |

Technical Note: Dynamic letters requiring movement tracing (`J` and `Z`) and heavily occluded letters (`E`, `P`, `Q`) are excluded from this static baseline to preserve high classification accuracy (>89%).

---

## 4. System Requirements

### Hardware Requirements

- Processor: Intel Core i3 / AMD Ryzen 3 or higher.
- Memory (RAM): Minimum 8GB (16GB recommended).
- Camera: Built-in or external USB webcam supporting 720p resolution (1280x720).
- Graphics Card: None required (operates on standard CPU).

### Software Requirements

- Operating System: Windows 10/11, macOS, or Ubuntu Linux.
- Web Browser: Google Chrome, Microsoft Edge, or any Chromium-based browser.
- Runtime Environment: Node.js (version 18 or higher).

---

## 5. Installation and Setup Guide (Step-by-Step)

Follow these steps in order. No prior programming background is required.

### Step 1: Install Node.js

Node.js is the software runtime that executes the application.
1. Open your browser and navigate to: https://nodejs.org/
2. Locate the download section and select the **Windows Installer (.msi)** under the LTS (Long Term Support) release.
3. Open the downloaded installer file.
4. Click **Next** on each screen, accept the license agreement, and click **Install**. Keep all settings at their defaults.
5. Once installation finishes, click **Finish**.

### Step 2: Prepare the Project Folder

1. Download or extract the project source code to a simple folder path on your computer (for example: `C:\asl-project` or `D:\asl-project`).
2. Important: Ensure the folder name contains only standard letters and hyphens. Do not use special characters such as `&`, `#`, or accented characters in the folder path.

### Step 3: Open the Terminal in the Project Folder

On Windows:

1. Open the project folder in File Explorer (where files such as `package.json` and `server.ts` are located).
2. Click directly on the address bar at the top of the File Explorer window.
3. Type `cmd` and press **Enter**.
4. A black Command Prompt window will appear, already pointed to your project folder.

### Step 4: Install Dependencies

In the Command Prompt window, type the following command exactly as shown and press **Enter**:

```bash
npm install --legacy-peer-deps
```

Wait 1 to 2 minutes while the computer downloads the necessary software packages. Once finished, the command prompt will return to normal without error messages.

---

## 6. How to Run the Application

### Option A: Web Application (Primary Recommended Method)

1. In the same Command Prompt window, type:

```bash
npm run dev
```

2. The terminal will display output confirming the server status:

```text
[ASL SignLens Server] Running on http://localhost:3000
```

3. Open Google Chrome or Microsoft Edge and navigate to:

```text
http://localhost:3000
```

4. A dialog will appear asking for camera access. Click **Allow**.
5. The webcam feed will initialize, and a neon tracking skeleton will display over your hand.

To stop the server at any time: Return to the Command Prompt window and press **Ctrl + C**.

### Option B: Standalone Desktop Pipeline (Python Alternative)

If you prefer running via Python rather than the web browser:

1. **Install Python**: Install **Python 3.9, 3.10, 3.11, or 3.12** from: https://www.python.org/
**Important**: Check the box **"Add Python to PATH"** during installation.
**Notice**: Do **NOT** use Python 3.13 or newer, as Google MediaPipe does not yet support Python 3.13+.

3. In your project directory, open Command Prompt and install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the desktop application:

```bash
python app.py
```

4. An OpenCV desktop window will open displaying the webcam feed and detection cards.

---

## 7. Controls and Usage Guide

### Signing and Detection

1. Position your hand 0.5 to 1.5 meters away from the webcam under adequate room lighting.
2. Form any of the 21 supported hand postures facing the camera.
3. Keep your hand steady for approximately 0.3 seconds (12 consecutive frames). When the confidence exceeds 80%, a confirmation sound will play and the letter will appear in the text buffer.
4. After each letter, a 1.0-second cooldown is enforced to allow you to transition your hand to the next sign without repeated inputs.

### Word Separation (Space)

- Hold all 5 fingers open facing the camera for **5.0 continuous seconds**.
- The progress indicator will fill up, and a `[SPACE]` token will be added to the buffer to separate words.

### Keyboard Shortcuts

- **Spacebar**: Sends the accumulated buffer to Gemini Flash Lite to synthesize a complete sentence and speak it aloud via Text-to-Speech.
- **C**: Clears the entire text buffer.
- **Backspace**: Deletes the most recently added character.

---

## 8. Gemini API Configuration (Optional)

The application includes an integrated **Offline Fallback Dictionary**. Common acronyms and words (such as `HI`, `ILY`, `MOM`, `RUN`, `THANK`, `DAY`, `FOOD`) will translate into complete, natural sentences even without an internet connection or API key.

If you wish to enable full generative sentence reconstruction for arbitrary letter combinations:

1. Visit Google AI Studio at: https://aistudio.google.com/apikey
2. Sign in with your Google account and click **Create API key**.
3. Copy the generated key string.
4. In the root project folder (`asl-project`), create a file named `.env`.
5. Open `.env` in Notepad and add the following line:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

6. Save the file and restart the application (`npm run dev`).

---

## 9. Troubleshooting and Common Errors

### Error: "'-gemini-pipeline\node_modules.bin' is not recognized"

- **Cause**: The project folder path contains the `&` symbol (for example: `project-&-pipeline`). On Windows, `&` is a reserved command operator that splits paths.
- **Fix**: Close Command Prompt, rename the folder to a simple name such as `asl-project`, and run `npm run dev` again.

### Error: "ERESOLVE could not resolve dependency"

- **Cause**: Version mismatch during package installation.
- **Fix**: Run `npm install --legacy-peer-deps` instead of standard `npm install`.

### Camera is blank or displays "Webcam Access Error"

- **Fix 1**: Ensure no other application (Zoom, Microsoft Teams, Skype) is currently using the camera.
- **Fix 2**: In Chrome/Edge, click the lock icon to the left of the URL bar (`http://localhost:3000`), ensure **Camera** is set to **Allow**, and refresh the page.
- **Fix 3**: In Windows Settings > Privacy & Security > Camera, verify that "Let desktop apps access your camera" is toggled **On**.

---

## 10. Project Information

- **Institution**: University of Science - VNU-HCM
- **Faculty**: Faculty of Information Technology
- **Course**: CSC00004 - Introduction to Information Technology
- **Assignment**: Project 2 - Developing an AI Application
- **License**: For academic assessment and educational purposes only (CSC00004 - VNU-HCM).
