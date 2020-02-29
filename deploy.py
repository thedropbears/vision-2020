from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
import sys

files = ["magic_numbers.py", "connection.py", "camera_manager.py", "utilities"]

try:
    for i, arg in enumerate(sys.argv[1:]):
        if arg[-9:] == "deploy.py":
            deploy_index = i

    if sys.argv[deploy_index + 1] == "loading-bay":
        main_file = "loading_bay_vision.py"
        print("Deploying Loading Bay code")

    elif sys.argv[deploy_index + 1] == "power-port":
        main_file = "power_port_vision.py"
        print("Deploying Power Port code")

    else:
        quit()
except:
    print(
        "Please ensure that you have `loading-bay` or `power-port` as the first argument"
    )
    quit()


server_ip = "10.47.74.11"
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
