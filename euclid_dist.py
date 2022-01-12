'''
Investigating the Euclidian distances on Moonshot data.
'''

import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

'''
Removing the acid fragment to reveal just the amine.
    Args:
        df (dataframe): The pre-split dataframe.
'''
def remove_acid(df):
    # The acid fragment smiles.
    acid = 'Clc1cc2c([C@H](C(Nc(cnc3)c4c3cccc4)=O)CN(CC=O)C2)cc1'
    acid = Chem.MolFromSmarts(acid)

    smiles = df['SMILES']
    peptides = [Chem.MolFromSmiles(s) for s in smiles]

    # Removing the acid fragment from the peptides.
    amines = []
    for mol in peptides:
        branch = AllChem.DeleteSubstructs(mol, acid)
        branch.UpdatePropertyCache()
        Chem.GetSymmSSSR(branch)
        amines.append(branch)

    df['Amine'] = amines
    return df

def main():
    # Importing the data.
    df = pd.read_csv('/Users/emmaking-smith/Moonshot/noisy_amides.csv', index_col=0)
    df = df.reset_index(drop=True)

    # Adding Inhibition (100 - Activity) column.
    df['Inhibition'] = 100 - df['Mean activity (%)']

    # Generating the amine fragments column.
    df = remove_acid(df)

    # Looking at the euclidian distances. Choose point 0 as reference.
    smiles = df['SMILES'].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [Chem.RDKFingerprint(m) for m in mols]

    df['fingerprint'] = fps

    amines = df['Amine'].to_list()
    amine_fps = [Chem.RDKFingerprint(m) for m in amines]

    df['amine_fps'] = amine_fps
    dists_fps = []
    dists_inhib = []

    random.seed(1)
    indices = random.sample(np.arange(len(df)).tolist(), 10)
    for n in indices:
        print(n)
        dists_amine_fps = []

        # Calculating the Euclidian distances between first entry and all other entries
        for i in range(len(df)):
            dists_fps.append( np.linalg.norm( np.array( df.loc[n, 'fingerprint'] - np.array(df.loc[i, 'fingerprint']) ) ) )
            dists_inhib.append( np.linalg.norm( df.loc[n, 'Inhibition'] - df.loc[i, 'Inhibition'] ) )
            dists_amine_fps.append(np.linalg.norm( np.array( df.loc[n, 'amine_fps'] ) - np.array(df.loc[i, 'amine_fps']) ) )

        # Plotting the distances between fingerprints vs distances.
        fig, ax = plt.subplots()
        ax.set_xlabel('Euclidian Amine Only Fingerprint Distance wrt Point ' + str(n))
        ax.set_ylabel('Inhibition')
        ax.set_title('Euclidian Distances of Amine Fingerprints Correlation')
        plt.scatter(dists_amine_fps, df['Inhibition'])
        plt.savefig('euclid_dist_amine_fps_' + str(n) + '.png')