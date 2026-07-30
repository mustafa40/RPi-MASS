import time
import serial


class ArduinoController:
    """Raspberry Pi ile Arduino arasındaki seri haberleşmeyi yönetir."""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud_rate: int = 9600
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.serial_connection = None

    def connect(self) -> bool:
        try:
            self.serial_connection = serial.Serial(
                self.port,
                self.baud_rate,
                timeout=1
            )

            time.sleep(2)
            print(f"Arduino bağlandı: {self.port}")
            return True

        except serial.SerialException as error:
            print(f"Arduino bağlantı hatası: {error}")
            return False

    def send_command(self, command: str) -> bool:
        if self.serial_connection is None:
            print("Arduino bağlantısı kurulmamış.")
            return False

        try:
            command = command.strip().upper()
            self.serial_connection.write(
                f"{command}\n".encode("utf-8")
            )

            print(f"Gönderilen komut: {command}")
            return True

        except serial.SerialException as error:
            print(f"Komut gönderme hatası: {error}")
            return False

    def normal(self) -> bool:
        return self.send_command("NORMAL")

    def fatigue(self) -> bool:
        return self.send_command("YORGUN")

    def tense(self) -> bool:
        return self.send_command("GERGIN")

    def stressed(self) -> bool:
        return self.send_command("STRESLI")

    def pump_test(self) -> bool:
        return self.send_command("POMPA")

    def fan_test(self) -> bool:
        return self.send_command("FAN")

    def turn_off(self) -> bool:
        return self.send_command("KAPAT")

    def close(self) -> None:
        if self.serial_connection is not None:
            self.turn_off()
            time.sleep(0.1)
            self.serial_connection.close()
            self.serial_connection = None
            print("Arduino bağlantısı kapatıldı.")
