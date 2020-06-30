import yaml
import pprint
import os

class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

Loader.add_constructor('!include', Loader.include)


with open("config_multispeaker_gst.yaml", 'r') as stream:
    try:
        y = yaml.load(stream, Loader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(y)
    except yaml.YAMLError as exc:
        print(exc)