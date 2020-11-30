"""
Authors: ST & MM
Model class implementation
"""

import os
import re

from Helpers.config import *


class Model:
    def __init__(self, model_type, path=None):
        """
        :param model_type: model type
        :param name: model name
        """
        self.type = model_type
        self.path = path

    def get_model_entry_point(self, model_path):
        """
        get the model entry point used to interact with the model
        :param model_path: model path
        :return: model entry point
        """
        for model_entry_point in os.listdir(model_path):
            if not model_entry_point.startswith('.'):
                name_from_file = re.search(r'^([^.]+)', model_entry_point).group(0)
                if self.name == name_from_file:
                    return model_entry_point
        raise FileNotFoundError('The model doesn\'t exist')
