import subprocess

file = "exp1.py"
iter = 10

for i in range(iter):
    subprocess.run(["python3",file])