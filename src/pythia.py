from psiK import eval_and_gen


classifier = eval_and_gen("psiK_1000.root")
if classifier is not None:
    eval_and_gen("pythia.root", classifier)
