import uproot
from typing import Dict
from uproot import TBranch


class InitialModel:
    def __init__(self, filename: str, *args, **kwargs):
        self.filename = filename
        self.args = args
        self.kwargs = kwargs

    def get_names(self):
        with uproot.open(self.filename) as file:
            tree: Dict[str, TBranch] = file["tuple/tuple;1"]
            print(tree.keys())

    def generate_data_mapping(self):
        with uproot.open(self.filename) as file:
            tree: Dict[str, TBranch] = file["tuple/tuple;1"]
            for event_n in tree["nBremPhotons"].array():
                print(event_n)
                pass


im = InitialModel("psiK_1000.root")
im.get_names()
# im.generate_data_mapping()
