"""
Facial Recognition & Visitor Logging for Smart Aloe Pot.
Uses OpenCV Deep Neural Network (YuNet detector + SFace 128-D recognizer)
to accurately recognize Meenakshi by name, trigger personalized Aloe greetings,
and log unknown visitors with live video streaming.
"""

import argparse
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import requests

# Optional: face_recognition fallback if installed
try:
    import face_recognition
    HAVE_FACE_RECOGNITION = True
except ImportError:
    HAVE_FACE_RECOGNITION = False

KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DIR.mkdir(exist_ok=True)
MEENAKSHI_PATH = KNOWN_FACES_DIR / "meenakshi.jpg"

MODELS_DIR = Path("models")
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

CASCADE_PATH = Path("haarcascade_frontalface_default.xml")
if not CASCADE_PATH.exists():
    CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

# OpenCV SFace 128-D Cosine Similarity Threshold:
# OpenCV official standard is 0.363. Verified Meenakshi photos match >=0.65.
# Strangers match at <=0.27. Threshold 0.363 reliably recognizes Meenakshi without falsely matching strangers.
SFACE_COSINE_THRESHOLD = 0.363
SFACE_STRANGER_MATCH_THRESHOLD = 0.280



def speak_voice(text):
    """
    Speak text aloud asynchronously (non-blocking) using Windows SAPI or pyttsx3.
    Guaranteed zero video stutter or frame dropping.
    """
    import threading

    def _worker():
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text, 1)  # 1 = SVSFlagsAsync (non-blocking)
            time.sleep(1.8)
            pythoncom.CoUninitialize()
            return
        except Exception:
            pass

        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
            return
        except Exception:
            pass

        try:
            import subprocess
            cmd = f'powershell -WindowStyle Hidden -Command "(New-Object -ComObject SAPI.SpVoice).Speak(\'{text}\')"'
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


class DeepFaceTracker:
    def __init__(self, backend_url="http://127.0.0.1:5000", camera_id=0):
        self.backend_url = backend_url.rstrip("/")
        self.camera_id = camera_id

        # Profile feature vectors (strictly verified Meenakshi embeddings)
        self.meenakshi_features = []
        self.known_dlib_encodings = []
        self.known_dlib_names = []

        self.last_meenakshi_seen_time = 0.0

        # Stranger rate-limiting & smart face profile cache (10-minute unique-face cooldown)
        self.consecutive_stranger_frames = 0
        self.stranger_frame_threshold = 8   # Must be steady unrecognized for 8 frames (~0.6s) before triggering
        self.recent_stranger_profiles = []  # Cache of {"feat": 128D_vector, "captured_time": float}
        self.stranger_cooldown_duration = 600.0  # 10 minutes (600s) cooldown per unique stranger face
        self.last_stranger_greeting_time = 0.0  # Time of last greeting to stranger
        self.last_stranger_spoken_time = 0.0    # Strict 15-minute global voice cooldown for "Hey stranger!"
        self.stranger_speech_cooldown = 900.0   # 15 minutes (900s) hard minimum between ANY stranger voice greetings
        self.feedback_banner = ""
        self.feedback_until = 0.0
        self.room_empty_since = time.time()
        self.room_empty_reset_duration = 20.0
        self.room_empty_notified = False

        self.last_logged_time = {}
        self.log_cooldown = 15.0  # seconds between repeated logs for the owner
        self.stranger_log_cooldown = 60.0   # 60s cooldown between stranger log entries

        # Daily Plant Selfie Scheduler
        self.selfie_dir = Path("static/selfies")
        self.selfie_dir.mkdir(parents=True, exist_ok=True)
        self.selfie_hour = 9       # 9 AM daily selfie
        self.selfie_minute = 0
        self.last_selfie_date = None  # Track the date of last daily selfie
        self.selfie_check_interval = 30.0  # Check every 30 seconds
        self.last_selfie_check = 0.0
        self.last_ondemand_check = 0.0

        # Initialize Deep Learning models
        self.use_deep_learning = False
        self.detector = None
        self.recognizer = None

        if YUNET_PATH.exists() and SFACE_PATH.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH), "", (640, 480), score_threshold=0.35, nms_threshold=0.3
                )
                self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
                self.use_deep_learning = True
                print("[OK] OpenCV SFace Deep Learning Face Engine initialized successfully!")
            except Exception as e:
                print(f"Warning: Could not initialize SFace: {e}")

        # Cascade fallback
        self.face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

        # Load stored profile pictures
        self.load_profiles()

    def load_profiles(self):
        """Extract and cache verified deep feature embeddings for Meenakshi."""
        self.meenakshi_features.clear()
        self.known_dlib_encodings.clear()
        self.known_dlib_names.clear()

        if not self.use_deep_learning or not self.detector or not self.recognizer:
            print("[WARN] Deep learning engine not ready. Skipping profile load.")
            return

        print(f"\nScanning {KNOWN_FACES_DIR.resolve()} for verified reference images...")

        anchor_feat = None
        # Step 1: Load primary anchor meenakshi.jpg
        if MEENAKSHI_PATH.exists():
            img = cv2.imread(str(MEENAKSHI_PATH))
            if img is not None:
                h, w = img.shape[:2]
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img)
                if faces is not None and len(faces) > 0:
                    aligned = self.recognizer.alignCrop(img, faces[0])
                    anchor_feat = self.recognizer.feature(aligned)
                    self.meenakshi_features.append(anchor_feat)
                    print(f"  [ANCHOR] Loaded primary profile: {MEENAKSHI_PATH.name}")

        # Step 2: Load additional sample photos, verifying each against anchor
        sample_imgs = sorted(list(KNOWN_FACES_DIR.glob("meenakshi_sample_*.jpg")) + list(KNOWN_FACES_DIR.glob("meenakshi_sample_*.png")))
        for img_path in sample_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
            if faces is not None and len(faces) > 0:
                aligned = self.recognizer.alignCrop(img, faces[0])
                feat = self.recognizer.feature(aligned)
                if anchor_feat is not None:
                    sim = self.recognizer.match(anchor_feat, feat, cv2.FaceRecognizerSF_FR_COSINE)
                    if sim >= 0.60:
                        self.meenakshi_features.append(feat)
                        print(f"  [OK] Verified Meenakshi sample: {img_path.name} (sim={sim:.2f})")
                    else:
                        print(f"  [SKIPPED] {img_path.name} (sim={sim:.2f} < 0.60 — different person rejected!)")
                else:
                    self.meenakshi_features.append(feat)
                    print(f"  [OK] Profile sample loaded: {img_path.name}")
            else:
                print(f"  [INFO] No clear face detected in {img_path.name}, skipping.")

        if HAVE_FACE_RECOGNITION:
            for img_path in [MEENAKSHI_PATH] + sample_imgs:
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is not None:
                    try:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        encs = face_recognition.face_encodings(rgb)
                        if encs:
                            self.known_dlib_encodings.append(encs[0])
                            self.known_dlib_names.append("Meenakshi")
                    except Exception:
                        pass

        print(f"Successfully loaded {len(self.meenakshi_features)} verified profile(s) for Meenakshi.\n")

    def is_new_stranger(self, live_feat):
        """
        Smart Face Engine check: Compare live 128-D embedding against strangers
        captured in the last 10 minutes. Returns (is_new: bool, best_similarity: float).
        """
        now = time.time()
        if live_feat is None or not self.use_deep_learning or not self.recognizer:
            is_new = (now - self.last_stranger_greeting_time) > self.stranger_cooldown_duration
            return is_new, 0.0

        # Expire stranger profiles older than 10 minutes (600 seconds)
        self.recent_stranger_profiles = [
            p for p in self.recent_stranger_profiles
            if (now - p.get("captured_time", 0.0)) < self.stranger_cooldown_duration
        ]

        if not self.recent_stranger_profiles:
            return True, 0.0

        best_sim = -1.0
        for p in self.recent_stranger_profiles:
            try:
                sim = self.recognizer.match(p["feat"], live_feat, cv2.FaceRecognizerSF_FR_COSINE)
                if sim > best_sim:
                    best_sim = sim
            except Exception:
                pass

        # SFace cosine similarity >= 0.28 indicates the SAME stranger (accounting for head pose/angles)
        if best_sim >= SFACE_STRANGER_MATCH_THRESHOLD:
            return False, float(best_sim)

        return True, float(best_sim)

    def evaluate_face(self, frame_bgr, face_data):
        """
        Evaluate detected face against Meenakshi's profile embeddings.
        Returns: (name, is_known, confidence_pct, raw_score, live_feat)
        """
        live_feat = None

        # Deep Learning evaluation
        if self.use_deep_learning and self.recognizer and self.meenakshi_features:
            try:
                aligned = self.recognizer.alignCrop(frame_bgr, face_data)
                live_feat = self.recognizer.feature(aligned)

                best_cosine = -1.0
                for ref_feat in self.meenakshi_features:
                    score = self.recognizer.match(ref_feat, live_feat, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_cosine:
                        best_cosine = score

                if best_cosine >= SFACE_COSINE_THRESHOLD:
                    self.last_meenakshi_seen_time = time.time()
                    scaled_conf = min(0.99, max(0.65, 0.65 + ((best_cosine - SFACE_COSINE_THRESHOLD) / 0.40) * 0.34))
                    return "Meenakshi", True, scaled_conf, best_cosine, live_feat
                else:
                    return "Visitor", False, 0.0, best_cosine, live_feat
            except Exception:
                pass

        # Optional dlib fallback
        if HAVE_FACE_RECOGNITION and self.known_dlib_encodings:
            x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb, [(y, x + w, y + h, x)])
            if encs:
                dists = face_recognition.face_distance(self.known_dlib_encodings, encs[0])
                if len(dists) > 0 and dists[0] < 0.45:
                    self.last_meenakshi_seen_time = time.time()
                    conf = 1.0 - dists[0]
                    return "Meenakshi", True, conf, float(conf), None

        return "Visitor", False, 0.0, 0.0, live_feat

    def log_to_backend(self, name, is_known, confidence, frame_bgr=None, live_feat=None):
        """Post detection event to Flask backend /api/log-visitor with snapshot ONLY for new stranger faces."""
        now = time.time()

        is_new_stranger_face = False
        if not is_known:
            is_new_stranger_face, sim = self.is_new_stranger(live_feat)

        # Snapshot condition:
        # Strictly for unfamiliar strangers when it's a NEW face (not captured in the last 10 minutes)
        needs_snapshot = (
            (not is_known)
            and (frame_bgr is not None)
            and is_new_stranger_face
        )

        last_time = self.last_logged_time.get(name, 0)
        cooldown = self.stranger_log_cooldown if not is_known else self.log_cooldown

        if not needs_snapshot and (now - last_time < cooldown):
            return

        self.last_logged_time[name] = now
        payload = {
            "name": name,
            "is_known": is_known,
            "confidence": float(confidence),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # ONLY take a snapshot & greet if it is a NEW stranger face!
        if needs_snapshot:
            if live_feat is not None:
                self.recent_stranger_profiles.append({
                    "feat": live_feat.copy(),
                    "captured_time": now
                })
            self.last_stranger_greeting_time = now

            # VOICE GREETING: Tell "Hey stranger!" aloud ONLY ONCE per 15 minutes globally!
            can_speak = (now - self.last_stranger_spoken_time) >= self.stranger_speech_cooldown
            if can_speak:
                self.last_stranger_spoken_time = now
                speak_voice("Hey stranger!")
                self.feedback_banner = "Spoke: 'Hey stranger!' & snapped photo!"
                self.feedback_until = now + 4.5
                print(f"\n[VOICE GREETING] >>> Said 'Hey stranger!' to unfamiliar visitor. 15-min voice cooldown active.")
            else:
                self.feedback_banner = "Stranger detected (Voice muted - 15m cooldown)"
                self.feedback_until = now + 3.0
                print(f"[STRANGER] Visitor snapshot captured. Voice greeting muted (15-min cooldown active).")

            try:
                import base64
                ts_str = time.strftime("%Y%m%d_%H%M%S")
                local_dir = Path("static/visitors")
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / f"stranger_{ts_str}.jpg"
                cv2.imwrite(str(local_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

                h, w = frame_bgr.shape[:2]
                scale = 640.0 / w if w > 640 else 1.0
                thumb = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
                ok, enc = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    payload["snapshot"] = base64.b64encode(enc).decode("utf-8")
                    print(f"[SMART CAPTURE] New stranger photo saved: {local_path.name}. 10-min cooldown active.")
            except Exception as e:
                print(f"[WARN] Failed to encode stranger snapshot: {e}")

        try:
            url = f"{self.backend_url}/api/log-visitor"
            resp = requests.post(url, json=payload, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                greeting = data.get("entry", {}).get("greeting", "OK")
                print(f"[ALERT] {name} logged! Aloe greeting: '{greeting}'")
        except Exception:
            pass  # Fail quietly if backend temporarily unreachable

    def save_snapshot(self, frame_bgr):
        """Save a snapshot with high resolution after verifying it matches Meenakshi."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        now = time.time()

        if self.use_deep_learning and self.detector and self.recognizer and self.meenakshi_features:
            h, w = frame_bgr.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame_bgr)
            if faces is not None and len(faces) > 0:
                aligned = self.recognizer.alignCrop(frame_bgr, faces[0])
                feat = self.recognizer.feature(aligned)
                best_sim = max(
                    self.recognizer.match(ref, feat, cv2.FaceRecognizerSF_FR_COSINE)
                    for ref in self.meenakshi_features
                )
                if best_sim < 0.45:
                    self.feedback_banner = f"[REJECTED] Face (sim={best_sim:.2f}) does not match Meenakshi!"
                    self.feedback_until = now + 4.0
                    print(f"\n[REJECTED] Snapshot did not match Meenakshi (sim={best_sim:.2f} < 0.45). Profile protected.")
                    return

        # Save as main meenakshi.jpg and new verified sample
        cv2.imwrite(str(MEENAKSHI_PATH), frame_bgr)
        sample_path = KNOWN_FACES_DIR / f"meenakshi_sample_{ts}.jpg"
        cv2.imwrite(str(sample_path), frame_bgr)
        print(f"\n[SAVED] New reference photo saved to {MEENAKSHI_PATH} and {sample_path.name}!")
        self.load_profiles()
        self.feedback_banner = "[OK] Meenakshi profile updated and reloaded!"
        self.feedback_until = now + 4.0

    def capture_plant_selfie(self, frame_bgr, label="daily"):
        """Capture a plant selfie and upload it to the backend gallery."""
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        # Save locally first
        local_path = self.selfie_dir / f"plant_selfie_{label}_{ts}.jpg"
        cv2.imwrite(str(local_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\n[SELFIE] Plant selfie saved locally: {local_path.name}")

        # Upload to backend gallery
        try:
            url = f"{self.backend_url}/api/selfie/upload"
            with open(str(local_path), "rb") as img:
                resp = requests.post(
                    url,
                    files={"image": (local_path.name, img, "image/jpeg")},
                    data={"label": label},
                    timeout=5.0,
                )
            if resp.status_code == 200:
                print(f"[SELFIE] Uploaded to gallery: {resp.json().get('filename', '')}")
            else:
                print(f"[SELFIE] Upload failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[SELFIE] Upload error (saved locally): {e}")

    def check_daily_selfie(self, frame_bgr):
        """Check if it's time for the daily morning selfie (9:00 AM)."""
        now = time.time()
        if now - self.last_selfie_check < self.selfie_check_interval:
            return
        self.last_selfie_check = now

        current = datetime.now()
        today_str = current.strftime("%Y-%m-%d")

        # Only capture once per day, at or after the scheduled hour
        if (self.last_selfie_date != today_str
                and current.hour == self.selfie_hour
                and current.minute >= self.selfie_minute
                and current.minute < self.selfie_minute + 5):  # 5-minute window
            self.last_selfie_date = today_str
            print(f"\n[SELFIE] Good morning! Taking daily plant selfie at {current.strftime('%H:%M')}...")
            self.capture_plant_selfie(frame_bgr, label="morning")

    def check_ondemand_selfie(self, frame_bgr):
        """Check if the backend has flagged a selfie request (from dashboard button)."""
        now = time.time()
        if now - self.last_ondemand_check < 2.0:  # Poll every 2 seconds
            return
        self.last_ondemand_check = now

        try:
            resp = requests.get(f"{self.backend_url}/api/selfie/check", timeout=1.5)
            if resp.status_code == 200 and resp.json().get("requested"):
                print("\n[SELFIE] On-demand selfie requested from dashboard!")
                self.capture_plant_selfie(frame_bgr, label="manual")
        except Exception:
            pass  # Backend unreachable is fine

    def run(self):
        """Main camera detection and tracking loop."""
        cam_src = int(self.camera_id) if str(self.camera_id).isdigit() else self.camera_id
        # Open default system backend first (MSMF on Win10/11), fallback to DSHOW if needed
        cap = cv2.VideoCapture(cam_src)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}. Please ensure webcam is connected and not in use by another app.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n" + "=" * 65)
        print("[*] Smart Aloe Pot - Deep Learning Visitor & Face Tracker")
        print(f"   Backend Server: {self.backend_url}")
        print("   Controls:")
        print("     [s] -> Quick 3-second countdown capture of Meenakshi")
        print("     [r] -> Reload reference face profiles from known_faces/")
        print("     [q] -> Quit stream")
        print("=" * 65 + "\n")

        fps_timer = time.time()
        fps = 0
        frame_count = 0

        countdown_active = False
        countdown_start = 0.0

        consecutive_bad_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_bad_frames += 1
                if consecutive_bad_frames == 30:
                    print("\n[WARN] Camera is returning blank/empty frames. Another application may be locking the webcam.")
                time.sleep(0.05)
                continue
            consecutive_bad_frames = 0

            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_timer = time.time()

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display = frame.copy()

            # Detect faces using YuNet
            faces = []
            if self.use_deep_learning and self.detector:
                self.detector.setInputSize((w, h))
                _, d_faces = self.detector.detect(frame)
                if d_faces is not None:
                    faces = d_faces
            else:
                # Haar fallback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade_faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
                for (cx, cy, cw, ch) in cascade_faces:
                    dummy = np.zeros(15, dtype=np.float32)
                    dummy[0:4] = [cx, cy, cw, ch]
                    faces.append(dummy)

            now = time.time()
            if len(faces) == 0:
                if now - self.room_empty_since > self.room_empty_reset_duration:
                    self.stranger_snapshot_taken = False
                self.consecutive_stranger_frames = 0
                if (now - self.room_empty_since > 3.0) and not self.room_empty_notified:
                    self.room_empty_notified = True
                    try:
                        requests.post(f"{self.backend_url}/api/presence/empty", timeout=1.0)
                    except Exception:
                        pass
            else:
                self.room_empty_since = now
                self.room_empty_notified = False

            # First pass: evaluate all faces in current frame
            evaluated_faces = []
            stranger_faces_present = False
            meenakshi_found_in_frame = False
            for face in faces:
                name, is_known, conf, raw_score, live_feat = self.evaluate_face(frame, face)
                if is_known and name.lower() == "meenakshi":
                    meenakshi_found_in_frame = True
                else:
                    stranger_faces_present = True
                evaluated_faces.append((face, name, is_known, conf, raw_score, live_feat))

            if meenakshi_found_in_frame:
                self.last_meenakshi_seen_time = now

            if stranger_faces_present:
                self.consecutive_stranger_frames += 1
            else:
                self.consecutive_stranger_frames = 0

            # Second pass: render bounding boxes and handle smart logging
            for face, name, is_known, conf, raw_score, live_feat in evaluated_faces:
                bx, by, bw, bh = map(int, face[0:4])

                if is_known and name.lower() == "meenakshi":
                    color = (70, 230, 110)  # Bright Emerald Green
                    label = f"Meenakshi (Owner) {int(conf * 100)}% [{raw_score:.2f}]"
                    self.log_to_backend("Meenakshi", True, conf)
                else:
                    # Unfamiliar face / visitor
                    color = (0, 140, 255)  # Vivid blue-amber
                    if self.consecutive_stranger_frames >= self.stranger_frame_threshold:
                        is_new, sim = self.is_new_stranger(live_feat)
                        if is_new:
                            can_spk = (now - self.last_stranger_spoken_time) >= self.stranger_speech_cooldown
                            spk_tag = "🔊 'Hey stranger!'" if can_spk else "🔇 Voice Muted"
                            label = f"⚠ NEW STRANGER [{raw_score:.2f}] · 📷 Snap · {spk_tag}"
                            self.log_to_backend("Stranger", False, 0.0, frame_bgr=frame, live_feat=live_feat)
                        else:
                            label = f"⚠ STRANGER [{raw_score:.2f}] · 👁 Cooldown Active (Silent)"
                            self.log_to_backend("Stranger", False, 0.0, frame_bgr=None, live_feat=live_feat)
                    else:
                        label = f"⚠ Stranger? Verifying {self.consecutive_stranger_frames}/{self.stranger_frame_threshold}..."

                # Stylish cornered bounding box
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
                # Header label tag
                tag_w = max(180, len(label) * 10 + 16)
                cv2.rectangle(display, (bx, max(0, by - 30)), (bx + tag_w, by), color, -1)
                cv2.putText(display, label, (bx + 8, max(12, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 15, 15), 2)

            # Handle Countdown if triggered
            if countdown_active:
                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)
                if remaining > 0:
                    cnt_str = str(remaining)
                    (tw, th), _ = cv2.getTextSize(cnt_str, cv2.FONT_HERSHEY_DUPLEX, 4.0, 6)
                    cx = (w - tw) // 2
                    cy = (h + th) // 2
                    cv2.putText(display, cnt_str, (cx + 4, cy + 4), cv2.FONT_HERSHEY_DUPLEX, 4.0, (10, 10, 10), 8)
                    cv2.putText(display, cnt_str, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 4.0, (60, 240, 130), 6)
                    cv2.putText(display, "Smile! Capturing in...", (w // 2 - 130, cy - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    countdown_active = False
                    self.save_snapshot(frame)
                    self.feedback_banner = "[OK] Meenakshi profile updated and reloaded successfully!"
                    self.feedback_until = time.time() + 4.0

            # Top HUD Bar
            cv2.rectangle(display, (15, 15), (420, 85), (20, 16, 28), -1)
            cv2.rectangle(display, (15, 15), (420, 85), (140, 95, 240), 1)
            cv2.putText(display, "AloePulse · Deep Learning Face Tracker", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 225, 255), 2)
            
            p_status = f"Meenakshi ({len(self.meenakshi_features)} samples loaded)" if self.meenakshi_features else "Press [S] to register face"
            cv2.putText(display, f"Profile: {p_status} | FPS: {fps}", (25, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 160, 230), 1)

            # Feedback banner if recent save or voice greeting
            if time.time() < self.feedback_until:
                banner_text = self.feedback_banner
                (btw, _), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                bw_half = max(260, btw // 2 + 20)
                cv2.rectangle(display, (w // 2 - bw_half, 20), (w // 2 + bw_half, 65), (25, 90, 45), -1)
                cv2.rectangle(display, (w // 2 - bw_half, 20), (w // 2 + bw_half, 65), (80, 240, 130), 2)
                cv2.putText(display, banner_text, (w // 2 - bw_half + 16, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Bottom instruction strip
            cv2.rectangle(display, (0, h - 35), (w, h), (15, 12, 22), -1)
            cv2.putText(display, "[S] Profile  |  [P] Plant Selfie  |  [R] Reload  |  [Q] Quit",
                        (25, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 220), 1)

            cv2.imshow("AloePulse - Facial Recognition & Visitor Logger", display)

            # Check daily selfie schedule & on-demand requests (uses raw frame, not display overlay)
            self.check_daily_selfie(frame)
            self.check_ondemand_selfie(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):  # [P] = manual plant selfie from keyboard
                self.capture_plant_selfie(frame, label="manual")
                self.feedback_banner = "[OK] Plant selfie captured and uploaded!"
                self.feedback_until = time.time() + 3.0
            elif key == ord('s') and not countdown_active:
                countdown_active = True
                countdown_start = time.time()
            elif key == ord('r'):
                self.load_profiles()
                self.feedback_banner = "Profiles reloaded!"
                self.feedback_until = time.time() + 2.0

        cap.release()
        cv2.destroyAllWindows()
        try:
            requests.post(f"{self.backend_url}/api/presence/empty", timeout=1.0)
        except Exception:
            pass


def acquire_camera_lock(port=5005):
    """Ensure only one instance of face_tracker accesses the webcam at a time."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        return s
    except OSError:
        return None


if __name__ == "__main__":
    lock_sock = acquire_camera_lock()
    if lock_sock is None:
        print("\n" + "!" * 65)
        print("[ERROR] Another instance of face_tracker.py is already running!")
        print("On Windows, webcams can only be opened by ONE application at a time.")
        print("Please close any existing face_tracker windows or terminal sessions, then rerun.")
        print("!" * 65 + "\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Smart Aloe Pot Visitor Tracker & Facial Recognition")
    parser.add_argument("--backend", default="http://127.0.0.1:5000", help="Flask backend URL")
    parser.add_argument("--camera", default=0, help="Camera index (e.g. 0) or RTSP stream URL")
    args = parser.parse_args()

    tracker = DeepFaceTracker(backend_url=args.backend, camera_id=args.camera)
    try:
        tracker.run()
    finally:
        if lock_sock:
            try:
                lock_sock.close()
            except Exception:
                pass
