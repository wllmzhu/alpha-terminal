import argparse
from src.algo_strategy import AlgoStrategy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-learning', action='store_true', default=False)
    parser.add_argument('--is-enemy', action='store_true', default=False)
    args = parser.parse_args()

    algo = AlgoStrategy(args)
    algo.start()