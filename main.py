from arduino_controller import ArduinoController


def main() -> None:
    arduino = ArduinoController()

    if not arduino.connect():
        return

    print("\nRPi-MASS kontrol sistemi")
    print("Komutlar:")
    print("NORMAL, YORGUN, GERGIN, STRESLI")
    print("POMPA, FAN, KAPAT, CIKIS\n")

    try:
        while True:
            command = input("Komut: ").strip().upper()

            if command == "CIKIS":
                break

            arduino.send_command(command)

    except KeyboardInterrupt:
        print("\nProgram durduruldu.")

    finally:
        arduino.close()


if __name__ == "__main__":
    main()
