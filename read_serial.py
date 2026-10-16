import serial
import serial.tools.list_ports
import time

def find_esp32():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "CH340" in port.description or "CP210" in port.description or "Serial" in port.description:
            return port.device
    
    # Fallback to just the first available port if we can't identify it
    if ports:
        return ports[0].device
    return None

port_name = find_esp32()
if not port_name:
    print("Could not find any connected ESP32 boards.")
    print("Make sure it's plugged in with a DATA-capable USB cable!")
    exit(1)

print(f"Opening {port_name} at 115200 baud...")

try:
    with serial.Serial(port_name, 115200, timeout=1) as ser:
        print("Successfully connected! Waiting for data from ESP32...\n" + "-"*40)
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error reading from serial: {e}")
                time.sleep(1)
except serial.SerialException as e:
    print(f"\nERROR: Could not open {port_name}.")
    print("This means another program is holding the port hostage!")
    print("1. Close the Arduino IDE completely.")
    print("2. Close any 3D printer software (like Cura/PrusaSlicer).")
    print("3. Unplug the ESP32 and plug it back in.")
