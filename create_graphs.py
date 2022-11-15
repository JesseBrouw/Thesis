import os
# subprocess is used to write bash cmommands
import subprocess

# script that loops through the directories in classical-domains.
if __name__ == '__main__':
    planning_dir = os.getcwd() + '/classical-domains/classical'
    
    for subdir, dirs, files in os.walk(planning_dir):
        for dir in dirs:
            print(f'Directory {dir} has the following files : ')
            for file in os.listdir(planning_dir + '/' + dir):
                print(file)


