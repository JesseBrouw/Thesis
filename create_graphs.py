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
            process = subprocess.check_call(['cp', filepath, './'])

            process = subprocess.check_call(['sudo', 'singularity', 'run', '-H', '$(pwd)', '-C', domain_file, problem, 'x.txt'])

            process = subprocess.check_call(['mkdir', '{}_{}_{}'.format(target_domain, domain_file.split(".")[0], problem.split(".")[0])])
            process = subprocess.check_call(['mv', 'abstract-structure-graph.txt', './{}_{}_{}'.format(target_domain, domain_file.split(".")[0], problem.split(".")[0])])
            process = subprocess.check_call(['mv', 'symmetry-graph.txt', './{}_{}_{}'.format(target_domain, domain_file.split(".")[0], problem.split(".")[0])])

            process = subprocess.check_call(['rm', problem])
            process = subprocess.check_call(['rm', domain_file])

    else:
        print('Domain {} has multiple domain files!'.format())


        




