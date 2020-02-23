"""The Connection class for The Drop Bears' vision code"""

import socket
import time

from typing import Tuple
from typing_extensions import Protocol

from networktables import NetworkTablesInstance

NetworkTables = NetworkTablesInstance.getDefault()

RIO_IP = "10.47.74.2"
UDP_RECV_PORT = 5005
UDP_SEND_PORT = 5006

Results = Tuple[float, float, float]


class Connection(Protocol):
    def send_results(self, results: Results) -> None:
        ...

    def pong(self) -> None:
        ...


class NTConnection:
    def __init__(self, inst: NetworkTablesInstance = NetworkTables) -> None:
        inst.initialize(server=RIO_IP)
        self.inst = inst

        nt = inst.getTable("/vision")
        self.entry = nt.getEntry("data")
        self.ping = nt.getEntry("ping")
        self.raspi_pong = nt.getEntry("raspi_pong")
        self.rio_pong = nt.getEntry("rio_pong")

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


class UDPConnection:
    def __init__(self) -> None:
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_send.bind(("255.255.255.255", UDP_SEND_PORT))
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(("0.0.0.0", UDP_RECV_PORT))

    def send_results(self, results: Results) -> None:
        self.sock_send.send(f"{results[0].hex()},{results[1].hex()}".encode())

    def pong(self) -> None:
        ...


class DummyConnection:
    def send_results(self, results: Results) -> None:
        ...

    def pong(self) -> None:
        pass
