import network
import urequests
import socket
import time
from machine import Pin, ADC

# Set the GPIO pins for relays
relay_solar = Pin(26, Pin.OUT)
relay_grid = Pin(27, Pin.OUT)
relay_battery = Pin(25, Pin.OUT)

# Set the GPIO pins for voltage and current sensors
voltage_sensor = ADC(Pin(34))
current_sensor = ADC(Pin(35))
voltage_sensor.atten(ADC.ATTN_11DB)
current_sensor.atten(ADC.ATTN_11DB)

# Define Access Point parameters
ssid = 'ESP32_Network'  # The name of the network created by ESP32
password = '123456789'  # Password for the network (optional)

# Flask API endpoint
flask_url = "http://<FLASK_SERVER_IP>:5000/predict"  # Replace with your Flask server IP

# Start the Access Point (AP) mode
def start_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=ssid, password=password)  # Set SSID and password for the AP
    while not ap.active():
        time.sleep(0.5)
        print("Waiting for AP to start...")
    print(f"Access Point '{ssid}' started.")
    print(f"IP address: {ap.ifconfig()[0]}")  # Display IP address of ESP32

# Read sensor data
def read_sensors():
    voltage = voltage_sensor.read() * (3.3 / 4095)  # Convert raw ADC values to voltage
    current = current_sensor.read() * (3.3 / 4095)  # Convert raw ADC values to current
    return round(voltage, 3), round(current, 3)

# Set relay based on received decision
def set_relay(source):
    relay_solar.off()
    relay_grid.off()
    relay_battery.off()
    if source == 'solar':
        relay_solar.on()
    elif source == 'grid':
        relay_grid.on()
    elif source == 'battery':
        relay_battery.on()

# Send sensor data to the Flask server and get the prediction
def get_prediction(voltage, current):
    weather_data = {
        'voltage': voltage,
        'current': current
    }
    
    try:
        response = urequests.post(flask_url, json=weather_data)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            print(f"Prediction: {prediction}")
            return prediction
        else:
            print("Failed to get prediction")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Serve a simple HTTP server to interact with the ESP32
def serve_requests():
    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    s = socket.socket()
    s.bind(addr)
    s.listen(1)
    print('Listening on', addr)

    while True:
        cl, addr = s.accept()
        print('Client connected from', addr)
        request = cl.recv(1024)
        print("Request:", request)
        
        voltage, current = read_sensors()
        print(f"Voltage: {voltage}V, Current: {current}A")
        
        # Get prediction from Flask server
        source = get_prediction(voltage, current)
        
        # Set the relay based on the prediction
        if source:
            set_relay(source)
        
        # Send HTTP response
        response = """HTTP/1.1 200 OK
Content-Type: text/html

<html>
  <head><title>ESP32 AP Mode</title></head>
  <body>
    <h1>ESP32 Access Point</h1>
    <p>Voltage: {}V</p>
    <p>Current: {}A</p>
    <p>Predicted Power Source: {}</p>
  </body>
</html>
""".format(voltage, current, source)

        cl.send(response)
        cl.close()

# Main program
start_ap()
serve_requests()
