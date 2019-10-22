# -*- coding: utf-8 -*-

import os

from CNNModel.config import config


with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
