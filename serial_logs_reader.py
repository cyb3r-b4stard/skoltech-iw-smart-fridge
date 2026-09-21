import serial
import serial.tools.list_ports
import time
import os

# --- Configuration ---
BAUD_RATE = 115200
OUTPUT_FILE = "banana_starts_to_spoil.csv"


def find_arduino_port():
    """Automatically search for an connected Arduino/Serial device."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Common identifiers for Arduino/CH340/FTDI/CP210x drivers
        if any(
            vendor in port.description.lower()
            for vendor in ["arduino", "ch340", "cp210", "ftdi", "usb serial"]
        ):
            return port.device
    
    # Fallback: if only one serial port is found, return it
    if len(ports) == 1:
        return ports[0].device
        
    return None


def start_logging():
    port = find_arduino_port()
    
    if not port:
        print("❌ Could not automatically detect Arduino.")
        port = input("Please manually enter your COM port (e.g., COM3 or /dev/ttyUSB0): ").strip()
    
    print(f"🔌 Connecting to {port} at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        # Give Arduino 2 seconds to reset upon serial connection
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"✅ Connected! Logging data directly to '{OUTPUT_FILE}'...")
        print("Press Ctrl+C to stop logging.\n")

        file_exists = os.path.isfile(OUTPUT_FILE)

        with open(OUTPUT_FILE, mode="a", encoding="utf-8") as csv_file:
            header_written = file_exists

            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()

                    if not line:
                        continue

                    # Handle CSV Header output from Arduino
                    if "timestamp_ms" in line:
                        if not header_written:
                            csv_file.write(line + "\n")
                            csv_file.flush()
                            header_written = True
                            print(f"📋 Header recorded: {line}")
                        continue

                    # Write measurement row
                    csv_file.write(line + "\n")
                    csv_file.flush()  # Ensure data is written to disk immediately
                    print(f"📥 Saved: {line}")

    except serial.SerialException as e:
        print(f"\n❌ Serial Communication Error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Logging stopped by user. File saved successfully.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    start_logging()
