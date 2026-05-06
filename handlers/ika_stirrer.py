import serial
import time


class IKAStirrer:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        time.sleep(2)  # wait for connection to stabilise
        print(f"Connected to IKA RCT digital on {port}")

    def send_command(self, command):
        """Send command to stirrer and return response."""
        full_command = command + ' \r\n'
        self.ser.write(full_command.encode())
        time.sleep(0.1)
        response = self.ser.readline().decode().strip()
        return response

    def set_speed(self, rpm):
        """Set stirrer speed in RPM."""
        response = self.send_command(f'OUT_SP_4 {rpm}')
        print(f"Speed set to {rpm} RPM")
        return response

    def start_stirring(self):
        """Start the stirrer motor."""
        response = self.send_command('START_4')
        print("Stirring started")
        return response

    def stop_stirring(self):
        """Stop the stirrer motor."""
        response = self.send_command('STOP_4')
        print("Stirring stopped")
        return response

    def get_actual_speed(self):
        """Read the actual current speed."""
        response = self.send_command('IN_PV_4')
        print(f"Actual speed: {response} RPM")
        return response

    def get_set_speed(self):
        """Read the set speed."""
        response = self.send_command('IN_SP_4')
        print(f"Set speed: {response} RPM")
        return response

    def disconnect(self):
        """Close serial connection."""
        self.ser.close()
        print("Disconnected from stirrer")


def main():
    stirrer = IKAStirrer(port='/dev/ttyACM0', baudrate=9600)
    try:
        stirrer.set_speed(1500)
        stirrer.start_stirring()
        print("Stirring at 1500 RPM for 30 seconds...")
        time.sleep(30)
        stirrer.get_actual_speed()
    finally:
        stirrer.stop_stirring()
        stirrer.disconnect()


if __name__ == "__main__":
    main()
