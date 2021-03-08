from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
import argparse
import time

files = ["magic_numbers.py", "connection.py", "camera_manager.py", "utilities", "vision_target.py", "collector.py", "balls_data.npz"]

parser = argparse.ArgumentParser(
    "Upload code to the Pi. Only one option (power-port or loading-bay) can be used. IP defaults to 10.47.74.11"
)
parser.add_argument(
    "-lb", "--loading-bay", help="Upload loading bay code", action="store_true"
)
parser.add_argument(
    "-pp", "--power-port", help="Upload power port code", action="store_true"
)
parser.add_argument("-bl", "--balls", help="Upload balls code", action="store_true")
parser.add_argument("-i", "--initial", help="Set pi to use Python", action="store_true")
parser.add_argument("-ip", "--ip", help="Specify a custom ip")
args = parser.parse_args()


if sum([args.loading_bay, args.power_port, args.balls]) > 1:
    print(parser.print_help())
    quit()

elif args.loading_bay:
    main_file = "loading_bay_vision.py"
    print("Deploying Loading Bay code")

elif args.power_port:
    main_file = "power_port_vision.py"
    print("Deploying Power Port code")

elif args.balls:
    main_file = "balls_vision.py"
    print("Deploying Balls code")

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
stdout, stdin, stderr = ssh.exec_command(
    "sudo mount -o remount,rw / ; sudo mount -o remount,rw /boot"
)
for line in stderr:
    print(line)
exit_status = stdout.channel.recv_exit_status()
if exit_status != 0:
    print(f"Something's gone wrong! Error exit status: {exit_status}")
    quit()
else:
    print("Done")

print("Uploading files ... ", end="")
scp = SCPClient(ssh.get_transport())
if args.initial:
    scp.put("runCamera")
    ssh.exec_command("chmod 755 runCamera")
scp.put(files, recursive=True)
scp.put(main_file, remote_path="~/uploaded.py")
print("Done")

print("Making file system read-only ... ", end="")
ssh.exec_command("sudo mount -o remount,ro / ; sudo mount -o remount,ro /boot")
print("Done")

print("Turning on vision ... ", end="")
ssh.exec_command("sudo svc -u /service/camera")
print("Done")

scp.close()
