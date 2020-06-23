import yaml
import pprint

with open("config_multispeaker_gst.yaml", 'r') as stream:
    try:
        y = yaml.safe_load(stream)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(y)
    except yaml.YAMLError as exc:
        print(exc)