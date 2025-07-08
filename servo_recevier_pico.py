import network
import socket
import time
from machine import Pin, PWM

# Wi-Fi credentials
ssid = 'Your WIFI name'
password = 'Your Wifi Passowrd' 

# Set up onboard LED
led = Pin("LED", Pin.OUT)

# Connect to Wi-Fi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(ssid, password)

print("Connecting to Wi-Fi", end="")
while not wlan.isconnected():
    time.sleep(0.5)
    print(".", end="")
print("\nConnected. IP:", wlan.ifconfig()[0])

# Setup servos
servo1 = PWM(Pin(15))
servo2 = PWM(Pin(14))
servo1.freq(50)
servo2.freq(50)

# Map angle to duty
def angle_to_duty(angle):
    min_duty = 1638
    max_duty = 8192
    return int(min_duty + (angle / 180.0) * (max_duty - min_duty))

# Socket setup
addr = socket.getaddrinfo("0.0.0.0", 80)[0][-1]
s = socket.socket()
s.bind(addr)
s.listen(1)

# For signal timeout detection
last_signal_time = time.time()
signal_timeout_sec = 5
error_state = False

print("Waiting for servo commands...")

while True:
    try:
        s.settimeout(0.5)  # non-blocking
        cl, addr = s.accept()
        request = cl.recv(1024).decode()
        error_state = False  # reset error flag

        # Parse servo commands
        if "GET /set?" in request:
            try:
                query = request.split("GET /set?")[1].split(" ")[0]
                params = query.split("&")
                for param in params:
                    if "servo1=" in param:
                        angle1 = int(param.split("=")[1])
                        servo1.duty_u16(angle_to_duty(angle1))
                    elif "servo2=" in param:
                        angle2 = int(param.split("=")[1])
                        servo2.duty_u16(angle_to_duty(angle2))

                # ✅ Valid signal received
                last_signal_time = time.time()
            except:
                error_state = True

        # Send response
        cl.send('HTTP/1.1 200 OK\r\nContent-type: text/plain\r\n\r\n')
        cl.send("OK")
        cl.close()

    except Exception as e:
        pass  # ignore timeout-based accept() errors

    # 🧠 Check if signal timeout occurred
    if error_state:
        # ⚠️ Blink LED to indicate error
        led.toggle()
        time.sleep(0.2)
    elif time.time() - last_signal_time > signal_timeout_sec:
        # ❌ No signal recently — turn off LED
        led.off()
    else:
        # ✅ Active communication — keep LED ON
        led.on()
