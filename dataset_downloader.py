from roboflow import Roboflow
from APIkeys import api_key
rf = Roboflow(api_key=api_key)
# project = rf.workspace("roboflow-100").project("poker-cards-cxcvz")
# version = project.version(2)
# dataset = version.download("tensorflow")


# project = rf.workspace("pocketsight-r1e4c").project("degen101-original-stack-size-rn0ra")
# version = project.version(12)
# dataset = version.download("tensorflow")

rf = Roboflow(api_key=api_key)
project = rf.workspace("devika-dl-onorl").project("project-1-sltff")
version = project.version(1)
dataset = version.download("tensorflow")


rf = Roboflow(api_key=api_key)
project = rf.workspace("ogar").project("dogs-vskxg")
version = project.version(5)
dataset = version.download("tensorflow")

