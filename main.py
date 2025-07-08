import customtkinter as ctk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
import requests
import time
import os
import shutil
from datetime import datetime
import random
import threading
import json

# === CONFIGURATION ===
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "USER_ID": "admin",
    "PASSWORD": "admin123"
}

INTRUDER_THRESHOLD = 85
VIDEO_SAVE_DIR = 'intruder_videos'
DATASET_DIR = 'dataset'
MODEL_FILE = 'face_trained.yml'
LABELS_FILE = 'people.npy'
PICO_IP = "http://10.241.149.19"
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 180
MOVE_THRESHOLD = 20
ANGLE_STEP_DIVISOR = 25
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FPS = 20
IMAGE_SIZE = (200, 200)


security_running = False
security_thread = None
settings_panel = None

# === Default Servo Calibration ===
CALIBRATION_FILE = "calibration.json"
DEFAULT_CALIBRATION = {"pan": 90, "tilt": 90}

def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(DEFAULT_CALIBRATION, f)
    with open(CALIBRATION_FILE, 'r') as f:
        return json.load(f)
    
calibration = load_calibration()
pan_angle = calibration["pan"]
tilt_angle = calibration["tilt"]

def save_calibration(pan, tilt):
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump({"pan": pan, "tilt": tilt}, f)

def calibrate_position():
    global pan_angle, tilt_angle
    msg = "Use W/A/S/D to adjust camera.\nW: Tilt Up | S: Tilt Down\nA: Pan Left | D: Pan Right\nR: Reset to 90,90\nQ: Save and Exit"
    messagebox.showinfo("Calibration Mode", msg)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Pan: {pan_angle}, Tilt: {tilt_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Calibrate Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save_calibration(pan_angle, tilt_angle)
            # Load and apply the saved calibration immediately
            calibration = load_calibration()
            pan_angle, tilt_angle = calibration["pan"], calibration["tilt"]
            break
        elif key == ord('r'):
            pan_angle, tilt_angle = 90, 90
        elif key == ord('a'):
            pan_angle = max(PAN_MIN, pan_angle - 2)
        elif key == ord('d'):
            pan_angle = min(PAN_MAX, pan_angle + 2)
        elif key == ord('w'):
            tilt_angle = max(TILT_MIN, tilt_angle - 2)
        elif key == ord('s'):
            tilt_angle = min(TILT_MAX, tilt_angle + 2)

        send_servo_angles(pan_angle, tilt_angle)

    cap.release()
    cv2.destroyAllWindows()

# === Load config ===
def load_config():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f)
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

config = load_config()
USER_ID = config.get("USER_ID", "admin")
PASSWORD = config.get("PASSWORD", "admin123")

def send_servo_angles(pan, tilt):
    try:
        requests.get(f"{PICO_IP}/set?servo1={pan}&servo2={tilt}", timeout=2)
    except:
        pass

def run_hardware_check():
    try:
        requests.get(f"{PICO_IP}/check", timeout=2)
    except:
        pass

def toggle_theme():
    if theme_var.get() == 1:
        ctk.set_appearance_mode("light")
        theme_switch.configure(text="☀")
    else:
        ctk.set_appearance_mode("dark")
        theme_switch.configure(text="\U0001F319")

def train_model():
    face_data = []
    labels = []
    label_map = []
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for idx, person in enumerate(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path): continue
        label_map.append(person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            face = cv2.resize(img, IMAGE_SIZE)
            face_data.append(face)
            labels.append(idx)
    if not face_data:
        messagebox.showwarning("Warning", "No face data found.")
        return
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(face_data, np.array(labels))
    model.save(MODEL_FILE)
    np.save(LABELS_FILE, np.array(label_map))
    messagebox.showinfo("Training", "Model training complete.")

def delete_member():
    members = [m for m in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, m))]
    if not members:
        messagebox.showinfo("Delete Member", "No members to delete.")
        return
    win = ctk.CTkToplevel()
    win.geometry("300x400")
    win.title("Delete Member")
    frame = ctk.CTkScrollableFrame(win)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    def delete(name):
        shutil.rmtree(os.path.join(DATASET_DIR, name))
        messagebox.showinfo("Deleted", f"Deleted member '{name}'. Retraining model...")
        win.destroy()
        train_model()
    for m in members:
        ctk.CTkButton(frame, text=m, command=lambda name=m: delete(name), fg_color="red").pack(pady=5, padx=10, fill='x')

def register_member():
    name = simpledialog.askstring("Register Member", "Enter member name:")
    if not name:
        return

    path = os.path.join(DATASET_DIR, name)
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible.")
        return

    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    directions = [
        ("Look Straight", "front"),
        ("Turn Head Left", "left"),
        ("Turn Head Right", "right")
    ]

    total_photos = 100
    images_per_direction = total_photos // len(directions)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for direction_text, dir_name in directions:
        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        count = 0
        while count < images_per_direction:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray, 1.1, 4)

            cv2.putText(frame, f"{direction_text}", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Images captured: {count}/{images_per_direction}", (10, 60), font, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Press 'Q' to cancel", (10, 450), font, 0.6, (0, 0, 255), 1)

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y + h, x:x + w], IMAGE_SIZE)
                file_name = os.path.join(dir_path, f"{count + 1}.jpg")
                cv2.imwrite(file_name, face_img)
                count += 1

            cv2.imshow("Face Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    train_model()
    messagebox.showinfo("Success", "Face registration complete and model retrained.")

def change_password():
    old = simpledialog.askstring("Change Password", "Enter current password:", show='*')
    if old != config.get("PASSWORD"):
        messagebox.showerror("Error", "Incorrect current password.")
        return
    new = simpledialog.askstring("Change Password", "Enter new password:", show='*')
    confirm = simpledialog.askstring("Change Password", "Confirm new password:", show='*')
    if not new or new != confirm:
        messagebox.showerror("Error", "Passwords do not match or are empty.")
    else:
        config["PASSWORD"] = new
        save_config(config)
        messagebox.showinfo("Success", "Password changed.")

def close_settings_panel(panel):
    panel.destroy()
    theme_switch.place(x=10, y=10)

def animate_settings_panel():
    global settings_panel
    theme_switch.place_forget()
    panel = ctk.CTkFrame(app, width=200, height=120)
    change_btn = ctk.CTkButton(panel, text="Change Password", command=change_password)
    close_btn = ctk.CTkButton(panel, text="Close", command=lambda: close_settings_panel(panel), width=80, height=25)
    change_btn.place(x=30, y=20)
    close_btn.place(x=60, y=70)
    panel.place(x=-200, y=50)
    for i in range(-200, 11, 10):
        panel.place(x=i, y=50)
        app.update()
        time.sleep(0.01)
    settings_panel = panel

def logout():
    main_frame.pack_forget()
    exit_btn.place_forget()
    login_frame.pack(pady=80)
    user_id_entry.delete(0, 'end')
    password_entry.delete(0, 'end')

def exit_app():
    logout()

def surveillance_loop():
    global pan_angle, tilt_angle, security_running
    calibration = load_calibration()
    default_pan, default_tilt = calibration["pan"], calibration["tilt"]
    pan_angle, tilt_angle = default_pan, default_tilt

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_FILE)
    labels = np.load(LABELS_FILE, allow_pickle=True)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
    run_hardware_check()
    send_servo_angles(pan_angle, tilt_angle)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access camera.")
        return

    recording = False
    last_intruder_time = 0
    last_detection_time = time.time()
    returned_to_default = False
    writer = None

    while security_running:
        ret, frame = cap.read()
        if not ret: break
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        faces = haar.detectMultiScale(gray, 1.3, 5)
        intruder_detected = False

        for (x, y, fw, fh) in faces:
            roi = cv2.resize(gray[y:y+fh, x:x+fw], IMAGE_SIZE)
            label, conf = model.predict(roi)
            name = labels[label] if conf < INTRUDER_THRESHOLD else "Intruder"
            color = (0,255,0) if name != "Intruder" else (0,0,255)
            if name == "Intruder":
                intruder_detected = True
                last_intruder_time = time.time()
                returned_to_default = False
                cx, cy = x + fw//2, y + fh//2
                dx, dy = cx - w//2, cy - h//2
                if abs(dx) > MOVE_THRESHOLD:
                    pan_angle += int(dx / ANGLE_STEP_DIVISOR)
                if abs(dy) > MOVE_THRESHOLD:
                    tilt_angle += int(dy / ANGLE_STEP_DIVISOR)
                pan_angle = max(PAN_MIN, min(PAN_MAX, pan_angle))
                tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle))
                send_servo_angles(pan_angle, tilt_angle)

            cv2.rectangle(frame, (x,y), (x+fw,y+fh), color, 2)
            cv2.putText(frame, f"{name} ({conf:.1f})", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if intruder_detected:
            if not recording:
                path = os.path.join(VIDEO_SAVE_DIR, datetime.now().strftime('%Y%m%d_%H%M%S') + '.avi')
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), FPS, (w, h))
                recording = True
        elif recording and time.time() - last_intruder_time > 3:
            recording = False
            if writer:
                writer.release()
                writer = None

        if recording and writer:
            writer.write(frame)

        # Return to default if no intruder for 5–8 seconds
        if not intruder_detected and time.time() - last_intruder_time > random.randint(5, 8) and not returned_to_default:
            pan_angle, tilt_angle = default_pan, default_tilt
            send_servo_angles(pan_angle, tilt_angle)
            returned_to_default = True

        cv2.imshow("Security", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    security_btn.configure(text="Start Security")
    security_running = False

def toggle_security():
    global security_running, security_thread
    if not security_running:
        if not os.path.exists(MODEL_FILE):
            messagebox.showerror("Error", "Model not trained yet.")
            return
        security_running = True
        security_btn.configure(text="Stop Security")
        security_thread = threading.Thread(target=surveillance_loop)
        security_thread.start()
    else:
        security_running = False

app = ctk.CTk()
app.title("Security UI")
app.geometry("500x600")

# Theme toggle
theme_var = ctk.IntVar(value=0)
theme_switch = ctk.CTkSwitch(app, variable=theme_var, command=toggle_theme, onvalue=1, offvalue=0, text="\U0001F319")
theme_switch.place(x=10, y=10)

# Login Frame
login_frame = ctk.CTkFrame(app, corner_radius=15)
login_frame.pack(pady=80)
user_id_entry = ctk.CTkEntry(login_frame, placeholder_text="User ID", width=220, corner_radius=10)
user_id_entry.pack(pady=10)
password_entry = ctk.CTkEntry(login_frame, placeholder_text="Password", show="*", width=220, corner_radius=10)
password_entry.pack(pady=10)
login_btn = ctk.CTkButton(login_frame, text="Login", command=lambda: handle_login(), width=150, corner_radius=15)
login_btn.pack(pady=20)

# Exit button (Logout) - initially hidden
exit_btn = ctk.CTkButton(app, text="❌", width=30, height=30, command=exit_app, fg_color="red")

# Main Panel
main_frame = ctk.CTkFrame(app, corner_radius=15)
ctk.CTkLabel(main_frame, text="Security Control Panel", font=("Arial", 20)).pack(pady=20)
security_btn = ctk.CTkButton(main_frame, text="Start Security", command=toggle_security, corner_radius=15)
security_btn.pack(pady=10)
ctk.CTkButton(main_frame, text="Register Member", command=register_member, corner_radius=15).pack(pady=10)
ctk.CTkButton(main_frame, text="Train Model", command=train_model, corner_radius=15).pack(pady=10)
ctk.CTkButton(main_frame, text="Delete Member", command=delete_member, corner_radius=15).pack(pady=10)
ctk.CTkButton(main_frame, text="Calibrate", command=calibrate_position, corner_radius=15).pack(pady=10)
ctk.CTkButton(main_frame, text="Settings", command=animate_settings_panel, corner_radius=15).pack(pady=10)

def handle_login():
    uid = user_id_entry.get()
    pwd = password_entry.get()
    if uid == config.get("USER_ID") and pwd == config.get("PASSWORD"):
        login_frame.pack_forget()
        exit_btn.place(x=460, y=10)
        main_frame.pack(pady=30)
    else:
        messagebox.showerror("Login Failed", "Incorrect credentials!")

app.bind('<Return>', lambda e: handle_login())

app.mainloop()
