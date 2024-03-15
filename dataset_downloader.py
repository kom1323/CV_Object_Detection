from roboflow import Roboflow
from APIkeys import api_key
rf = Roboflow(api_key=api_key)
project = rf.workspace("roboflow-100").project("poker-cards-cxcvz")
version = project.version(2)
dataset = version.download("tensorflow")


project = rf.workspace("zerocomputervision").project("chip-detection-and-counting-v2")
version = project.version(1)
dataset = version.download("tensorflow")