"""The Connection class for The Drop Bears' vision code"""

import time
from networktables import NetworkTables

PI_IP = "10.47.74.6"
RIO_IP = "10.47.74.2"
UDP_RECV_PORT = 5005
UDP_SEND_PORT = 5006
TIME_TO_PONG = 0.00000001


class Connection:
    def __init__(self, using_nt=False, test: bool = False):
        """Initialises Connection class.

        Args:
            using_nt (bool)
            entries (list): list of the names, in order, of the
            networktables entries (only if using_nt = True)
        """
        if test:
            self.test = True
        else:
            self.test = False
            self.using_nt = using_nt
            if self.using_nt:
                # self.entries = entries
                self.init_NT_connection()
            else:
                self.init_UDP_connection()

    def init_NT_connection(self):
        """Initialises NetworkTables connection to the RIO"""
        NetworkTables.initialize(server=RIO_IP)
        NetworkTables.setUpdateRate(1)
        self.nt = NetworkTables.getTable("/vision")
        # for i, entry in enumerate(self.entries):
        #   Replace entry strings with nt entries
        #    self.entries[i] = self.nt.getEntry(entry)
        self.entry = self.nt.getEntry("data")
        self.ping = self.nt.getEntry("ping")
        self.raspi_pong = self.nt.getEntry("raspi_pong")
        self.rio_pong = self.nt.getEntry("rio_pong")
        self.fps_entry = self.nt.getEntry("fps")

        self.old_ping_time = 0

    def init_UDP_connection(self):
        """Initialises UDP connection to the RIO"""
        import socket

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind((RIO_IP, UDP_RECV_PORT))

    def send_results(self, results):
        """Sends results to the RIO depending on connecion type. Returns Nothing."""
        if self.test:
            pass
        elif self.using_nt:

            self.entry.setDoubleArray(results)
            NetworkTables.flush()
        else:
            self.sock_send.sendto(
                f"{results[0]},{results[1]}".encode("utf-8"), (PI_IP, UDP_SEND_PORT)
            )

    def pong(self) -> None:
        self.ping_time = self.ping.getNumber(0)
        if abs(self.ping_time - self.old_ping_time) > TIME_TO_PONG:
            self.rio_pong.setNumber(self.ping_time)
            self.raspi_pong.setNumber(time.monotonic())
            self.old_ping_time = self.ping_time
