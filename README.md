# AI_BASED_HOME_SECURITY
AI-based surveillance system using OpenCV, MediaPipe, and CustomTkinter. Detects faces, tracks intruders via servo (Pico W), logs activity, supports camera calibration, and ignores known members. Ideal for smart security projects.
Features
    Real-time Face Detection & Recognition
    Intruder Detection & Tracking with servo motor using Wi-Fi (Raspberry Pi Pico W)
    AI Model Training on custom dataset with automatic retraining on updates
    Register Home Members with 300+ guided face images (left, right, front)
    Camera Calibration Mode with WASD control and persistent position
    Modern UI using customtkinter with emoji buttons, animations, and feedback
    Face tracking only for intruders — known members are ignored
    Intruder logs & auto image capture (10 images if intruder persists > 3 seconds)
    Returns to default position after 5–8 seconds of no detection
    Light/Dark mode with sun/moon toggle

Technologies Used
    Python
    OpenCV
    MediaPipe
    CustomTkinter
    Socket Programming (Wi-Fi communication with Raspberry Pi Pico W)
    NumPy, threading, JSON

Hardware Used
    Raspberry Pi Pico W (Wi-Fi + Servo Motor control)
    USB Camera
    Servo Motor (Pan/Tilt)
    Speakers for alert sound
