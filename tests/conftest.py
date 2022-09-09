import sys
import warnings
from os.path import abspath, dirname, join


git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)
