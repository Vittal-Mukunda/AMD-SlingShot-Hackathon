import sys
import pytest


def main(argv=None):
    argv = argv or ["-q"]
    return pytest.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
