from .scrnet import SCRNet
from .hscnet import HSCNet, HSCNetUnc

def get_model(name, dataset):
    return {
            'scrnet' : SCRNet(),
            'hscnet' : HSCNet(dataset=dataset),
            'hscnet_unc': HSCNetUnc(dataset=dataset)
           }[name]

