class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, module):
        self._module_dict[module.__name__] = module
        return module

    def get(self, name):
        return self._module_dict.get(name)

MODELS = Registry('models')
DATASETS = Registry('datasets')
LOSSES = Registry('losses')
METRICS = Registry('metrics')
