#!/bin/env python3
from argparse import ArgumentParser
import os


def main(prefix):
    parser = ArgumentParser()
    parser.add_argument("-m", "--map", type=str, default="MoveToBeacon",
                        help="map name")
    parser.add_argument("-r", "--rule_based", action="store_true",
                        help="whether using rule-based agent")
    parser.add_argument("-e", "--episodes", type=int, default=0,
                        help="max episodes")
    args = parser.parse_args()
    agent = "rl_midterm"
    agent += f".{'ruled_agent' if args.rule_based else 'rl_agent'}.{args.map}"

    cmd = [
        "cd", prefix, ";",
        "python3", "-m", "pysc2.bin.agent",
        "--agent", agent,
        "--map", args.map,
        "--use_feature_units",
        "--max_episodes", args.episodes
    ]

    os.system(' '.join(map(str, cmd)))


if __name__ == '__main__':
    main(os.path.dirname(__file__))
