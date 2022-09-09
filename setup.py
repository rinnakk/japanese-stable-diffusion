from setuptools import setup, find_packages
from codecs import open


def _requires_from_file(filename):
    return open(filename).read().splitlines()


exec(open('src/japanese_stable_diffusion/version.py').read())
setup(
    name="japanese_stable_diffusion",
    version=__version__,
    author="rinna Co., Ltd.",
    description="Japanese Stable Diffusion",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    url="https://github.com/rinnakk/japanese-stable-diffusion",
    long_description_content_type="text/markdown",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    license='Creative ML OpenRAIL-M License',
    install_requires=_requires_from_file('requirements.txt'),
    extras_require={'dev': ['pytest', 'python-dotenv']},
)
