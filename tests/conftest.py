import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_script(name, filename):
    """Import a script of the repository by path.

    test.py cannot be imported as "test": that name is the standard library's
    regression test package.
    """
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
