"""
Interactive Face Profile Capture Tool for Smart Aloe Pot.
Helps Meenakshi capture clean, high-quality reference photos with:
- On-screen framing oval guide and real-time lighting check
- 3-second countdown timer before taking the shot
- Instant preview before saving
- Direct integration with OpenCV SFace Deep Learning embeddings
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DIR.mkdir(exist_ok=True)
MEENAKSHI_PATH = KNOWN_FACES_DIR / "meenakshi.jpg"

MODELS_DIR = Path("models")
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def check_sharpness(gray_crop):
    """Compute Laplacian variance to detect blurry captures."""
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def check_brightness(gray_crop):
    """Compute mean brightness of face region."""
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    return float(np.mean(gray_crop))


def main():
    print("\n" + "=" * 65)
    print("[*] Aloe Pot - Meenakshi Profile Photo Studio")
    print("=" * 65)
    print("Instructions:")
    print("  1. Center your face inside the on-screen oval guide.")
    print("  2. Ensure good lighting on your face (avoid strong backlighting).")
    print("  3. Press [SPACE] or [S] to start the 3-second countdown.")
    print("  4. Press [Y] to confirm and save, or [R] to retake.")
    print("  5. Press [Q] to exit.")
    print("=" * 65 + "\n")

    # Load deep learning models
    detector = None
    recognizer = None
    if YUNET_PATH.exists() and SFACE_PATH.exists():
        try:
            detector = cv2.FaceDetectorYN.create(
                str(YUNET_PATH), "", (640, 480), score_threshold=0.6, nms_threshold=0.3
            )
            recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
            print("[OK] Deep Learning face recognition engine loaded (YuNet + SFace)!")
        except Exception as e:
            print(f"Warning: Could not initialize Deep Learning models: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam (index 0). Please check camera permissions.")
        return

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    countdown_active = False
    countdown_start = 0.0
    countdown_seconds = 3

    preview_active = False
    captured_frame = None
    captured_face_aligned = None
    captured_face_bbox = None
    captured_score = 0.0
    status_msg = "Center face in guide, then press [SPACE] to capture"
    status_color = (200, 220, 255)

    saved_count = len(list(KNOWN_FACES_DIR.glob("meenakshi*.jpg")))

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        # Update detector input size
        if detector is not None:
            detector.setInputSize((w, h))

        # Face detection
        detected_face = None
        face_bbox = None
        sharpness = 0.0
        brightness = 0.0

        if detector is not None:
            _, faces = detector.detect(frame)
            if faces is not None and len(faces) > 0:
                detected_face = faces[0]
                bx, by, bw, bh = map(int, detected_face[0:4])
                face_bbox = (bx, by, bw, bh)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                crop_g = gray[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
                sharpness = check_sharpness(crop_g)
                brightness = check_brightness(crop_g)

        # Draw framing guide oval in center
        guide_center = (w // 2, int(h * 0.48))
        guide_axes = (int(w * 0.16), int(h * 0.28))

        if not preview_active:
            # Check if face is centered
            is_centered = False
            if face_bbox:
                fx = face_bbox[0] + face_bbox[2] // 2
                fy = face_bbox[1] + face_bbox[3] // 2
                dist_center = np.hypot(fx - guide_center[0], fy - guide_center[1])
                is_centered = dist_center < (guide_axes[0] * 0.8)

            guide_color = (60, 220, 100) if is_centered else (120, 120, 160)
            cv2.ellipse(display, guide_center, guide_axes, 0, 0, 360, guide_color, 2, cv2.LINE_AA)

            # Draw face box if detected
            if face_bbox:
                bx, by, bw, bh = face_bbox
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), guide_color, 2)
                cv2.putText(display, "Face Detected", (bx, max(20, by - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)

            # Quality meter bar at top
            cv2.rectangle(display, (20, 20), (450, 100), (20, 16, 28), -1)
            cv2.rectangle(display, (20, 20), (450, 100), (140, 100, 240), 1)

            light_status = "Good" if 70 <= brightness <= 200 else ("Dim (Add light)" if brightness < 70 else "Too Bright")
            sharp_status = "Sharp" if sharpness > 80 else "A bit blurry (hold still)"
            
            cv2.putText(display, f"Lighting: {light_status} ({int(brightness)})", (35, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(display, f"Focus: {sharp_status}", (35, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(display, f"Saved Photos: {saved_count}", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 240, 150), 1)

            # Handle Countdown
            if countdown_active:
                elapsed = time.time() - countdown_start
                remaining = countdown_seconds - int(elapsed)

                if remaining > 0:
                    # Draw big countdown number in center
                    cnt_str = str(remaining)
                    (tw, th), _ = cv2.getTextSize(cnt_str, cv2.FONT_HERSHEY_DUPLEX, 4.0, 6)
                    cx = (w - tw) // 2
                    cy = (h + th) // 2
                    # Shadow
                    cv2.putText(display, cnt_str, (cx + 4, cy + 4), cv2.FONT_HERSHEY_DUPLEX, 4.0, (10, 10, 10), 8)
                    cv2.putText(display, cnt_str, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 4.0, (50, 230, 120), 6)
                    cv2.putText(display, "Hold still & smile!", (w // 2 - 120, cy + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    # Capture now!
                    countdown_active = False
                    preview_active = True
                    captured_frame = frame.copy()
                    if detected_face is not None and recognizer is not None:
                        captured_face_aligned = recognizer.alignCrop(frame, detected_face)
                        captured_face_bbox = face_bbox
                        status_msg = "Captured! Press [Y] to SAVE or [R] to RETAKE"
                        status_color = (60, 240, 120)
                    else:
                        status_msg = "No clear face found. Press [R] to try again"
                        status_color = (80, 80, 255)

            # Bottom instruction banner
            cv2.rectangle(display, (0, h - 50), (w, h), (15, 12, 22), -1)
            cv2.putText(display, status_msg, (25, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        else:
            # Preview mode: show captured image
            display = captured_frame.copy()
            # Dim background slightly
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

            # Highlight captured face
            if captured_face_bbox:
                bx, by, bw, bh = captured_face_bbox
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (60, 240, 120), 3)

            # Show aligned face thumbnail in corner if available
            if captured_face_aligned is not None:
                thumb = cv2.resize(captured_face_aligned, (160, 160))
                display[30:190, w - 190:w - 30] = thumb
                cv2.rectangle(display, (w - 190, 30), (w - 30, 190), (60, 240, 120), 2)
                cv2.putText(display, "Aligned Crop", (w - 185, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1)

            # Preview modal box
            cv2.rectangle(display, (w // 2 - 280, h - 120), (w // 2 + 280, h - 20), (20, 16, 28), -1)
            cv2.rectangle(display, (w // 2 - 280, h - 120), (w // 2 + 280, h - 20), (140, 100, 240), 2)
            cv2.putText(display, "Happy with this picture?", (w // 2 - 250, h - 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(display, "[Y] or [ENTER] = Save as Meenakshi    [R] = Retake", (w // 2 - 250, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 240, 160), 2)

        cv2.imshow("Meenakshi Profile Photo Studio - Smart Aloe Pot", display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in [ord(' '), ord('s')] and not preview_active and not countdown_active:
            countdown_active = True
            countdown_start = time.time()
            status_msg = "Get ready! Capturing in 3 seconds..."
            status_color = (100, 240, 255)
        elif preview_active:
            if key in [ord('y'), 13]:  # 'y' or ENTER
                # Save primary profile
                cv2.imwrite(str(MEENAKSHI_PATH), captured_frame)
                
                # Also save high-quality numbered sample for multi-sample matching
                sample_idx = len(list(KNOWN_FACES_DIR.glob("meenakshi_sample_*.jpg"))) + 1
                sample_path = KNOWN_FACES_DIR / f"meenakshi_sample_{sample_idx}.jpg"
                cv2.imwrite(str(sample_path), captured_frame)

                print("\n" + "=" * 60)
                print(f"[SUCCESS] Saved Meenakshi profile:")
                print(f"  Primary: {MEENAKSHI_PATH.resolve()}")
                print(f"  Sample #{sample_idx}: {sample_path.resolve()}")
                print("=" * 60 + "\n")

                saved_count += 1
                preview_active = False
                status_msg = f"[OK] Photo #{sample_idx} saved! Press [SPACE] to add another angle or [Q] to finish"
                status_color = (60, 255, 120)

            elif key == ord('r'):
                preview_active = False
                status_msg = "Retake: Center face and press [SPACE] when ready"
                status_color = (255, 220, 100)

    cap.release()
    cv2.destroyAllWindows()
    print("Photo Studio closed.")


if __name__ == "__main__":
    main()
