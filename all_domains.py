import os
import sys
# subprocess is used to write bash cmommands
import subprocess
import time

if __name__ == '__main__':
    domain_path = os.getcwd() + '/classical/'

    domains = os.listdir(domain_path)

    for domain in domains:
        process = subprocess.Popen(['python3', 'create_graphs.py', domain])
        process.wait()
