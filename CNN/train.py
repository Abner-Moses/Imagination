"""CNN family training entry point; implementation is shared in common."""

from __future__ import annotations
import argparse
from common.runner import train_family


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=("research", "small", "tiny"), default="research")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(train_family("cnn", args.seed, args.profile, args.force))


if __name__ == "__main__":
    main()
