# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import io
import os
import re
from setuptools import setup, find_packages

# This method was adapted from code in
#  https://github.com/albumentations-team/albumentations
def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "sgtapose", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(
    name='sgtapose',
    version=get_version(),
    author='Nimolty',
    author_email='2301110749@pku.edu.cn',
    maintainer='Nimolty',
    maintainer_email='2301110749@pku.edu.cn',
    description='Robot Structure Prior Guided Temporal Attention for Camera-to-Robot Pose Estimation from Image Sequence',
    packages=['sgtapose'],
    package_dir={'sgtapose': 'sgtapose'},
    zip_safe=False
)
