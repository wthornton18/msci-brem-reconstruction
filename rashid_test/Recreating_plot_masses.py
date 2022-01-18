import uproot as up
import matplotlib.pyplot as plt
import numpy as np
import vector
from tqdm import tqdm
import hist

plt.rcParams['axes.xmargin'] = 0

infilename = 'C:\\Users\Rashi\\Documents\\MSci_project\\rashid_test\\data\\Bu2JpsiK_ee_1000_events.root'
tree = up.open(infilename)['tuple/tuple']
branches = tree.arrays()
#trueK = vector.zip({'px':branches['K_PX'],'py':branches['K_PY'],'pz':branches['K_PZ'],'m':branches['K_M']})


hBmass_noreco  = []
hBmass_newreco = []
hBmass_stdreco = []


hJPsimass_noreco  = []
hJPsimass_newreco = []
hJPsimass_stdreco = []




for ievent in tqdm(range(1000)):

    branchesE = branches[2*ievent]

    trueK = vector.zip({'px':branchesE['K_PX'],'py':branchesE['K_PY'],'pz':branchesE['K_PZ'],'m':branchesE['K_M']})
    trueB = vector.zip({'px':branchesE['B_PX'],'py':branchesE['B_PY'],'pz':branchesE['B_PZ'],'m':branchesE['B_M']})
    trueJpsi = vector.zip({'px':branchesE['JPsi_PX'],'py':branchesE['JPsi_PY'],'pz':branchesE['JPsi_PZ'],'m':branchesE['JPsi_M']})

    if branchesE.nElectronTracks < 1 or branchesE.ElectronTrack_TYPE[0] != 3:
    #not a well reconstructed track
        continue

    eplus = vector.zip({'px':branchesE['ElectronTrack_PX'][0],'py':branchesE['ElectronTrack_PY'][0],'pz':branchesE['ElectronTrack_PZ'][0],'m':branchesE['electron_M']})
    eplus_newreco = vector.zip({'px':0,'py':0,'pz':0,'m':0})

    eplus_newreco = eplus_newreco + eplus
    for ibrem in range(branchesE.nBremPhotons):
        if branchesE.BremPhoton_OVZ[ibrem] > 5000:
            continue
        trueBrem = vector.zip({'px':branchesE.BremPhoton_PX[ibrem],'py':branchesE.BremPhoton_PY[ibrem],'pz':branchesE.BremPhoton_PZ[ibrem],'m':branchesE.BremPhoton_M[ibrem]})
        eplus_newreco = eplus_newreco + trueBrem
    
    eplus_stdreco = vector.zip({'px':branchesE.StdBremReco_Electron_PX, 'py':branchesE.StdBremReco_Electron_PY, 'pz':branchesE.StdBremReco_Electron_PZ, 'mass':branchesE.StdBremReco_Electron_M})

    branchesO = branches[2*ievent + 1]
    if branchesO.nElectronTracks < 1 or branchesO.ElectronTrack_TYPE[0] != 3:
    #not a well reconstructed track
        continue

    eminus = vector.zip({'px':branchesO['ElectronTrack_PX'][0],'py':branchesO['ElectronTrack_PY'][0],'pz':branchesO['ElectronTrack_PZ'][0],'m':branchesO['electron_M']})
    eminus_newreco = vector.zip({'px':0,'py':0,'pz':0,'m':0})
    
    eminus_newreco = eminus_newreco + eminus

    for ibrem in range(branchesO.nBremPhotons):
        if branchesO.BremPhoton_OVZ[ibrem] > 5000:
            continue
        trueBrem = vector.zip({'px':branchesO.BremPhoton_PX[ibrem],'py':branchesO.BremPhoton_PY[ibrem],'pz':branchesO.BremPhoton_PZ[ibrem],'m':branchesO.BremPhoton_M[ibrem]})
        eminus_newreco = eminus_newreco + trueBrem


    eminus_stdreco = vector.zip({'px':branchesO.StdBremReco_Electron_PX, 'py':branchesO.StdBremReco_Electron_PY, 'pz':branchesO.StdBremReco_Electron_PZ, 'mass':branchesO.StdBremReco_Electron_M})

    JPsi_noreco = eminus + eplus
    JPsi_newreco = eminus_newreco + eplus_newreco
    JPsi_stdreco = eminus_stdreco + eplus_stdreco

    B_noreco = eminus + eplus + trueK
    B_newreco = eminus_newreco + eplus_newreco + trueK
    B_stdreco = eminus_stdreco + eplus_stdreco + trueK

    hBmass_noreco.append(B_noreco.mass)
    hBmass_newreco.append(B_newreco.mass)
    hBmass_stdreco.append(B_stdreco.mass)

    hJPsimass_noreco.append(JPsi_noreco.mass)
    hJPsimass_newreco.append(JPsi_newreco.mass)
    hJPsimass_stdreco.append(JPsi_stdreco.mass)

hBmass_noreco = np.abs(np.array(hBmass_noreco))
hBmass_newreco = np.abs(np.array(hBmass_newreco))
hBmass_stdreco = np.abs(np.array(hBmass_stdreco))

hJPsimass_noreco = np.abs(np.array(hJPsimass_noreco))
hJPsimass_newreco = np.abs(np.array(hJPsimass_newreco))
hJPsimass_stdreco = np.abs(np.array(hBmass_stdreco))

# THIS IS DIFFERENT



plt.hist([hBmass_noreco,hBmass_newreco,hBmass_stdreco], bins = 50,histtype='step',stacked=True, fill=False, range =(3000,5600))
plt.hist([hJPsimass_noreco,hJPsimass_newreco, hJPsimass_stdreco], histtype='step',bins = 50,stacked=True, fill=False, range = (1000,3400))



# tree2 = up.open('uproot-tutorial-file.root:Events')
# branches2 = tree2.arrays()
# muon_p4 = vector.zip({'pt': branches2['Muon_pt'], 'eta': branches2['Muon_eta'], 'phi': branches2['Muon_phi'], 'mass': branches2['Muon_mass']})
# two_muons_mask = branches2['nMuon'] == 2
# two_muons_p4 = muon_p4[two_muons_mask]
# first_muon_p4 = two_muons_p4[:, 0]
# second_muon_p4 = two_muons_p4[:, 1]
# plt.hist(first_muon_p4.deltaR(second_muon_p4), bins=100)
# plt.xlabel('$\Delta R$ between muons')
# plt.ylabel('Number of two-muon events')
# plt.show()


