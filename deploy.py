from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
import argparse
import time

files = ["magic_numbers.py", "connection.py", "camera_manager.py", "utilities"]

parser = argparse.ArgumentParser(
    "Upload code to the Pi. Only one option (power-port or loading-bay) can be used. IP defaults to 10.47.74.11"
)
parser.add_argument(
    "-lb", "--loading-bay", help="Upload loading bay code", action="store_true"
)
parser.add_argument(
    "-pp", "--power-port", help="Upload power port code", action="store_true"
)
parser.add_argument("-ip", "--ip", help="Specify a custom ip")
args = parser.parse_args()


if args.loading_bay and args.power_port:
    print(parser.print_help())
    quit()

elif args.loading_bay:
    main_file = "loading_bay_vision.py"
    print("Deploying Loading Bay code")

elif args.power_port:
    main_file = "power_port_vision.py"
    print("Deploying Power Port code")

else:
    parser.print_help()
    quit()


server_ip = "10.47.74.11" if args.ip is None else args.ip
username = "pi"
password = "raspberry"

ssh = SSHClient()
ssh.set_missing_host_key_policy(AutoAddPolicy())
print(f"Connecting to the pi at {server_ip} ... ", end="")
ssh.connect(server_ip, username=username, password=password)
print("Done")

print("Turning off vision ... ", end="")
ssh.exec_command("sudo svc -d /service/camera")
print("Done")

print("Making file system writable ... ", end="")
ssh.exec_command("rw")
print("Done")

time.sleep(0.5)

print("Uploading files ... ", end="")
scp = SCPClient(ssh.get_transport())
scp.put(files, recursive=True)
scp.put(main_file, remote_path="~/uploaded.py")
print("Done")

print("Making file system read-only ... ", end="")
ssh.exec_command("ro")
print("Done")

print("Turning on vision ... ", end="")
ssh.exec_command("sudo svc -u /service/camera")
print("Done")

scp.close()