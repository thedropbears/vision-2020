import time

from networktables import NetworkTablesInstance
from typing import Tuple

NetworkTables = NetworkTablesInstance.getDefault()

RIO_IP = "10.47.74.2"
UDP_RECV_PORT = 5005
UDP_SEND_PORT = 5006

Results = Tuple[float, float, float]


class NTConnection:
    def __init__(self, inst: NetworkTablesInstance = NetworkTables) -> None:
        inst.initialize(server=RIO_IP)
        self.inst = inst

        nt = inst.getTable("/vision")
        self.entry = nt.getEntry("data")
        self.ping = nt.getEntry("ping")
        self.raspi_pong = nt.getEntry("raspi_pong")
        self.rio_pong = nt.getEntry("rio_pong")
        self.fps_entry = nt.getEntry("fps")

        self.old_fps_time = 0.0
        
        self.last_ping_time = 0.0
        self.time_to_pong = 0.00000001
        self._get_time = time.monotonic

    def send_results(self, results: Results) -> None:
        self.entry.setDoubleArray(results)
        self.inst.flush()

    def pong(self) -> None:
        self.ping_time = self.ping.getNumber(0)
        if abs(self.ping_time - self.last_ping_time) > self.time_to_pong:
            self.rio_pong.setNumber(self.ping_time)
            self.raspi_pong.setNumber(self._get_time())
            self.last_ping_time = self.ping_time

    def set_fps(self) -> None:
        current_time = time.monotonic()
        fps = 1/(current_time-self.old_fps_time)
        self.old_fps_time = current_time
        self.fps_entry.setDouble(fps)



class DummyConnection:
    def send_results(self, results: Results) -> None:
        print(results)

    def pong(self) -> None:
        ...
