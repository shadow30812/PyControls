import unittest


def run_tests():
    """
    Discover and run all tests in the 'tests' directory.
    """
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner(buffer=True)
    runner.run(suite)


if __name__ == "__main__":
    run_tests()
