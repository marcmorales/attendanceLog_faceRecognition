import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import face_recognition
import cv2
import time
import pickle
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "saved_encodings.pkl"
video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
cv_scaler = 4

# --- WINDOW NAMES ---
window_student = 'Student View'
window_teacher = 'Teacher View (Admin)'

# --- STATE VARIABLES ---
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
attendance_log = {}
ignored_encodings = []

running = True
button_rect = (0, 0, 0, 0)


def load_or_train_faces():
    if os.path.exists(ENCODINGS_FILE):
        print(f"Loading known faces from {ENCODINGS_FILE}...")
        with open(ENCODINGS_FILE, 'rb') as file:
            return pickle.load(file)

    print("No saved encodings found. Learning faces from the 'known_faces' directory...")
    known_encodings = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created '{KNOWN_FACES_DIR}' folder.")
        return [], []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"Learned face: {name}")

    if known_encodings:
        with open(ENCODINGS_FILE, 'wb') as file:
            pickle.dump((known_encodings, known_names), file)
            print(f"Saved all face data to {ENCODINGS_FILE}")

    return known_encodings, known_names


known_face_encodings, known_face_names = load_or_train_faces()


def get_name_from_gui(face_crop):
    window_name = "New Face Detected!"
    cv2.namedWindow(window_name)

    name_input = ""
    h, w = face_crop.shape[:2]

    canvas_w = max(w, 400)
    canvas_h = h + 120

    while True:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        x_offset = (canvas_w - w) // 2
        canvas[0:h, x_offset:x_offset + w] = face_crop

        cv2.putText(canvas, "Type name, press ENTER to save.", (15, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
        cv2.putText(canvas, "(Leave blank and press ENTER to ignore)", (15, h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 100, 100), 1)
        cv2.putText(canvas, f"> {name_input}_", (15, h + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(0)

        if key == 13 or key == 10:  # Enter
            break
        elif key == 8 or key == 127:  # Backspace
            name_input = name_input[:-1]
        elif 32 <= key <= 126:  # Printable chars
            name_input += chr(key)

    cv2.destroyWindow(window_name)
    return name_input.strip()


def process_frame(frame):
    global face_locations, face_names, face_encodings, attendance_log
    global known_face_encodings, known_face_names, ignored_encodings

    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

    face_names = []

    for i, face_encoding in enumerate(face_encodings):
        name = "Unknown"
        is_known = False

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
                is_known = True

                if name not in attendance_log:
                    current_time = datetime.now().strftime("%I:%M:%S %p")
                    attendance_log[name] = current_time
                    print(f"Logged Attendance: {name} at {current_time}")

        is_ignored = False
        if not is_known and len(ignored_encodings) > 0:
            ignore_distances = face_recognition.face_distance(ignored_encodings, face_encoding)
            if np.min(ignore_distances) < 0.6:
                is_ignored = True

        if not is_known and not is_ignored:
            top, right, bottom, left = face_locations[i]
            h, w = frame.shape[:2]

            pad = 40
            y1 = max(0, (top * cv_scaler) - pad)
            y2 = min(h, (bottom * cv_scaler) + pad)
            x1 = max(0, (left * cv_scaler) - pad)
            x2 = min(w, (right * cv_scaler) + pad)

            face_crop = frame[y1:y2, x1:x2]

            # --- Show exact waiting message on Student View ---
            student_wait_frame = frame.copy()
            overlay = student_wait_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, student_wait_frame, 0.4, 0, student_wait_frame)

            cv2.putText(student_wait_frame, "Please wait while instructor enters name",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(window_student, student_wait_frame)
            cv2.waitKey(1)  # Force the GUI to update immediately
            # --------------------------------------------------

            new_name = get_name_from_gui(face_crop)

            if new_name != "":
                name = new_name
                img_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.jpg")

                cv2.imwrite(img_path, face_crop)
                known_face_encodings.append(face_encoding)
                known_face_names.append(new_name)

                with open(ENCODINGS_FILE, 'wb') as file:
                    pickle.dump((known_face_encodings, known_face_names), file)

                print(f"Successfully added {new_name} to the database!")
                attendance_log[name] = datetime.now().strftime("%I:%M:%S %p")
            else:
                ignored_encodings.append(face_encoding)
                print("Ignored this person for the rest of the session.")

        face_names.append(name)

    return frame


def draw_results(frame):
    global button_rect

    student_view = frame.copy()
    recognized_names = []

    # Draw bounding boxes and names on the core frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw on Teacher's frame
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Draw box on Student's view
        cv2.rectangle(student_view, (left, top), (right, bottom), color, 2)

        if name != "Unknown" and name not in recognized_names:
            recognized_names.append(name)

    # --- Draw Welcome messages on Student View ---
    if recognized_names:
        h, w = student_view.shape[:2]
        # Create a dark background banner for readability
        overlay = student_view.copy()
        banner_height = 10 + (40 * len(recognized_names))
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, student_view, 0.4, 0, student_view)

        y_text = 35
        for rec_name in recognized_names:
            cv2.putText(student_view, f"Welcome, {rec_name}!", (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_text += 40
    # ---------------------------------------------

    # Create the Teacher View (annotated frame + sidebar)
    frame_h, frame_w = frame.shape[:2]
    sidebar_w = 400

    teacher_view = np.zeros((frame_h, frame_w + sidebar_w, 3), dtype=np.uint8)
    teacher_view[:, :frame_w] = frame
    teacher_view[:, frame_w:] = (35, 35, 35)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(teacher_view, "ATTENDANCE LOG", (frame_w + 20, 40), font, 1.0, (255, 255, 255), 2)
    cv2.line(teacher_view, (frame_w + 20, 55), (frame_w + sidebar_w - 20, 55), (255, 255, 255), 1)

    y_offset = 100
    for logged_name, time_in in attendance_log.items():
        cv2.putText(teacher_view, logged_name, (frame_w + 20, y_offset), font, 0.8, (0, 255, 0), 2)
        cv2.putText(teacher_view, time_in, (frame_w + 20, y_offset + 30), font, 0.6, (200, 200, 200), 1)
        y_offset += 70

    # Draw the Stop Button on the Teacher View
    bx1 = frame_w + 40
    by1 = frame_h - 100
    bx2 = frame_w + sidebar_w - 40
    by2 = frame_h - 40
    button_rect = (bx1, by1, bx2, by2)

    cv2.rectangle(teacher_view, (bx1, by1), (bx2, by2), (0, 0, 200), -1)
    cv2.rectangle(teacher_view, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
    cv2.putText(teacher_view, "EXIT SYSTEM", (bx1 + 75, by1 + 40), font, 0.8, (255, 255, 255), 2)

    return student_view, teacher_view


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


def handle_mouse_click(event, x, y, flags, param):
    global running
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = button_rect
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            print("Exit button clicked! Shutting down...")
            running = False


# --- MULTI-WINDOW SETUP ---
cv2.namedWindow(window_student, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_teacher, cv2.WINDOW_NORMAL)

# Attach our mouse listener ONLY to the teacher's window
cv2.setMouseCallback(window_teacher, handle_mouse_click)

# --- MAIN LOOP ---
while running:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    processed_frame = process_frame(frame)

    # Unpack the two separate views
    display_student, display_teacher = draw_results(processed_frame)

    current_fps = calculate_fps()

    # Add FPS counter to the Teacher's View
    cv2.putText(display_teacher, f"FPS: {current_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show both windows side-by-side
    cv2.imshow(window_student, display_student)
    cv2.imshow(window_teacher, display_teacher)

    if cv2.waitKey(1) == ord("q"):
        running = False

video_capture.release()
cv2.destroyAllWindows()