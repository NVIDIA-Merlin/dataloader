#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import itertools
import os
import sys

from setuptools import find_namespace_packages, setup

try:
    import versioneer
except ImportError:
    # we have a versioneer.py file living in the same directory as this file, but
    # if we're using pep 517/518 to build from pyproject.toml its not going to find it
    # https://github.com/python-versioneer/python-versioneer/issues/193#issue-408237852
    # make this work by adding this directory to the python path
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import versioneer


def read_requirements(filename):
    base = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(base, filename), "rb", "utf-8") as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]


requirements = {
    "base": read_requirements("requirements/base.txt"),
    "tensorflow": read_requirements("requirements/tensorflow.txt"),
    "pytorch": read_requirements("requirements/torch.txt"),
    "torch": read_requirements("requirements/torch.txt"),
    "jax": read_requirements("requirements/jax.txt"),
    "dev": read_requirements("requirements/dev.txt"),
}

with open("README.md", encoding="utf8") as readme:
    long_description = readme.read()

setup(
    name="merlin-dataloader",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_namespace_packages(include=["merlin*"]),
    url="https://github.com/NVIDIA-Merlin/dataloader",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements["base"],
    test_suite="tests",
    python_requires=">=3.8",
    extras_require={
        **requirements,
        "all": list(itertools.chain(*list(requirements.values()))),
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
)
