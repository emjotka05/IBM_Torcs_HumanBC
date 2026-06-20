import sys

from human_drive import human_collect_data


def get_laps(default=10):
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--laps="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return default
        if arg == "--laps" and i + 2 <= len(sys.argv[1:]):
            try:
                return int(sys.argv[i + 2])
            except ValueError:
                return default
    return default


if __name__ == "__main__":
    human_collect_data(
        num_laps=get_laps(10),
        auto_reset_each_lap=True,
    )
