import argparse
import ctypes
import struct
import time

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


MAGIC = bytes([0xA5, 0x5A, 0x10, 0x30])
ACK = b"\x06"


class WinSerial:
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    PURGE_TXCLEAR = 0x0004

    class DCB(ctypes.Structure):
        _fields_ = [
            ("DCBlength", ctypes.c_uint32),
            ("BaudRate", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("wReserved", ctypes.c_uint16),
            ("XonLim", ctypes.c_uint16),
            ("XoffLim", ctypes.c_uint16),
            ("ByteSize", ctypes.c_ubyte),
            ("Parity", ctypes.c_ubyte),
            ("StopBits", ctypes.c_ubyte),
            ("XonChar", ctypes.c_char),
            ("XoffChar", ctypes.c_char),
            ("ErrorChar", ctypes.c_char),
            ("EofChar", ctypes.c_char),
            ("EvtChar", ctypes.c_char),
            ("wReserved1", ctypes.c_uint16),
        ]

    class COMMTIMEOUTS(ctypes.Structure):
        _fields_ = [
            ("ReadIntervalTimeout", ctypes.c_uint32),
            ("ReadTotalTimeoutMultiplier", ctypes.c_uint32),
            ("ReadTotalTimeoutConstant", ctypes.c_uint32),
            ("WriteTotalTimeoutMultiplier", ctypes.c_uint32),
            ("WriteTotalTimeoutConstant", ctypes.c_uint32),
        ]

    def __init__(self, port, baud):
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        device = port if port.startswith("\\\\.\\") else "\\\\.\\" + port
        self.handle = self.kernel32.CreateFileW(
            device,
            self.GENERIC_READ | self.GENERIC_WRITE,
            0,
            None,
            self.OPEN_EXISTING,
            0,
            None,
        )
        if self.handle == self.INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())

        dcb = self.DCB()
        dcb.DCBlength = ctypes.sizeof(self.DCB)
        if not self.kernel32.GetCommState(self.handle, ctypes.byref(dcb)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

        dcb.BaudRate = baud
        dcb.flags = 1
        dcb.ByteSize = 8
        dcb.Parity = 0
        dcb.StopBits = 0
        if not self.kernel32.SetCommState(self.handle, ctypes.byref(dcb)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

        timeouts = self.COMMTIMEOUTS(0, 0, 0, 0, 3000)
        self.kernel32.SetCommTimeouts(self.handle, ctypes.byref(timeouts))
        self.kernel32.PurgeComm(self.handle, self.PURGE_TXCLEAR)

    def write(self, data):
        written = ctypes.c_uint32(0)
        buf = ctypes.create_string_buffer(data)
        ok = self.kernel32.WriteFile(
            self.handle,
            buf,
            len(data),
            ctypes.byref(written),
            None,
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return written.value

    def read(self, size=1):
        data = bytearray()
        for _ in range(size):
            buf = ctypes.create_string_buffer(1)
            read_count = ctypes.c_uint32(0)
            ok = self.kernel32.ReadFile(
                self.handle,
                buf,
                1,
                ctypes.byref(read_count),
                None,
            )
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            if read_count.value == 0:
                break
            data.extend(buf.raw[:1])
        return bytes(data)

    def close(self):
        if getattr(self, "handle", None):
            self.kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def open_serial_port(port, baud):
    if serial is not None and hasattr(serial, "Serial"):
        return serial.Serial(port, baud, timeout=1, write_timeout=3)
    return WinSerial(port, baud)


def frame_to_rgb565_le(frame_bgr):
    b = frame_bgr[:, :, 0].astype(np.uint16) >> 3
    g = frame_bgr[:, :, 1].astype(np.uint16) >> 2
    r = frame_bgr[:, :, 2].astype(np.uint16) >> 3
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565.astype("<u2").tobytes()


def main():
    parser = argparse.ArgumentParser(description="Send PC webcam frames to PZ103 TFTLCD over UART.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--no-ack", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    interval = 1.0 / args.fps if args.fps > 0 else 0
    header = MAGIC + struct.pack("<HH", args.width, args.height)

    with open_serial_port(args.port, args.baud) as ser:
        time.sleep(2.0)
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            payload = frame_to_rgb565_le(frame)
            ser.write(header)
            ser.write(payload)
            if not args.no_ack:
                ack = ser.read(1)
                if ack != ACK:
                    print("No ACK from board; check baud rate, wiring, and flashed firmware.")
                    time.sleep(0.5)
                    continue

            if args.preview:
                cv2.imshow("PZ103 camera stream", frame)
                if cv2.waitKey(1) == 27:
                    break

            elapsed = time.perf_counter() - start
            if interval > elapsed:
                time.sleep(interval - elapsed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
