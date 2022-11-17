import os
import sys
# subprocess is used to write bash cmommands
import subprocess
import time

# Script that targets one given domain an
if __name__ == '__main__':
    target_domain = sys.argv[1]
    domain_path = os.getcwd() + '/classical/' + target_domain

    files = os.listdir(domain_path)
    # filter out all the pddl files
    files = list(filter(lambda x: x.split(".")[1] == 'pddl', files))
    # filter out domain.pddl file (directories with 1 domain for all problems)
    is_domain = lambda x: x.split(".")[0] == "domain"
    one_domain = list(filter(is_domain, files))

    if one_domain:
        domain_file = 'domain.pddl'
        files.remove('domain.pddl')
        # copy domain input file to current directory (where singularity resides)
        process = subprocess.Popen(['cp', os.path.join(domain_path, domain_file), './'])
        process.wait()
        
        for problem in files:
            # copy problem file to current directory (where singularity resides)
            filepath = os.path.join(domain_path, problem)
            process = subprocess.Popen(['cp', filepath, './'])
            process.wait()

            print(domain_file, problem)
            process = subprocess.Popen(['sudo', 'singularity', 'run', '-H', os.getcwd(), '-C', 'graphs.sif', domain_file, problem, 'x.txt'])
            process.wait()
            time.sleep(1)
            process = subprocess.Popen(['mkdir', '{}_{}_{}'.format(target_domain, domain_file.split(".")[0], problem.split(".")[0])])
            process.wait()
            process = subprocess.Popen(['mv', 'abstract-structure-graph.txt', './{}_{}_{}'.format(target_domain, domain_file.split(".")[0], problem.split(".")[0])])
            process.wait()
            process = subprocess.Popen(['mv', 'symmetry-graph.txt', './{}_{}_{}'.format(target_domain, domain_file.split(".")[0], problem.split(".")[0])])
            process.wait()
            process = subprocess.Popen(['rm', problem])
            process.wait()
            process = subprocess.Popen(['rm', domain_file])
            process.wait()

    else:
        print('Domain {} has multiple domain files!'.format())


        




