from pyparsing import Dict
import uproot
from uproot import TBranch


def generate_energy_curve(filename: str):
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;4"]
        for e_cluster, e_brem in zip(tree["BremCluster_E"].array(), tree["BremPhoton_E"].array()):
            print(e_cluster)
            print(e_brem)


if __name__ == "__main__":
    generate_energy_curve("1000ev.root")
