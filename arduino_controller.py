#!/usr/bin/env python3
import glob
import time
import serial

class NucleoController:
    VALID_COMMANDS = {"NORMAL", "TENSE", "STRESSED", "FATIGUED", "OFF"}

    def __init__(self, port=None, baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None

    def _find_port(self):
        if self.port:
            return self.port
        ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        return ports[0] if ports else None

    def connect(self):
        if self.serial is not None and self.serial.is_open:
            return True
        port = self._find_port()
        if port is None:
            print("No Nucleo serial port found.")
            return False
        try:
            self.serial = serial.Serial(port=port, baudrate=self.baudrate,
                                        timeout=self.timeout, write_timeout=self.timeout)
            self.port = port
            time.sleep(2.0)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            print(f"Nucleo connected: {self.port}")
            return True
        except Exception as error:
            print(f"Nucleo connection failed: {error}")
            self.serial = None
            return False

    def send_command(self, command):
        command = str(command).strip().upper()
        if command not in self.VALID_COMMANDS:
            raise ValueError(f"Unsupported Nucleo command: {command}")
        if self.serial is None or not self.serial.is_open:
            if not self.connect():
                raise RuntimeError("Nucleo is not connected.")
        self.serial.write((command + "\n").encode("utf-8"))
        self.serial.flush()
        print(f"TX -> {command}")

    def normal(self): self.send_command("NORMAL")
    def tense(self): self.send_command("TENSE")
    def stressed(self): self.send_command("STRESSED")
    def fatigued(self): self.send_command("FATIGUED")
    def off(self): self.send_command("OFF")

    def close(self):
        if self.serial is not None:
            try:
                if self.serial.is_open:
                    self.serial.close()
            finally:
                self.serial = None
