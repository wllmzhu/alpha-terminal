import os
import subprocess
import sys
import argparse

# Runs a single game
def run_single_game(process_command):
    print("Start run a match")
    p = subprocess.Popen(
        process_command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
        )
    # daemon necessary so game shuts down if this script is shut down by user
    p.daemon = 1
    p.wait()
    print("Finished running match")

# Get location of this run file
file_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(file_dir, os.pardir)
parent_dir = os.path.abspath(parent_dir)

# Get if running in windows OS
is_windows = sys.platform.startswith('win')
print("Is windows: {}".format(is_windows))

# Set default path for algos if script is run with no params
my_algo = parent_dir + "/python-algo/run_and_learn.sh"
enemy_algo = parent_dir + "/python-algo/run_enemy.sh"

print("Algo 1: ", my_algo)
print("Algo 2: ", enemy_algo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-turns', type=int, default=False)
    args = parser.parse_args()

    for _ in range(args.n_turns):
        run_single_game("cd {} && java -jar engine.jar work {} {}".format(parent_dir, my_algo, enemy_algo))