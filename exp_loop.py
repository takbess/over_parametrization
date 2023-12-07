import subprocess

file = "main.py"
iter = 5

for i in range(iter):
    subprocess.run(["python3",file])