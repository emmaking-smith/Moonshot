'''
PCA Analysis on Moonshot data with scikit and umap.
'''

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

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

    pca = PCA(n_components=2)
    X = [np.array(x).tolist() for x in df['amine_fps']]
    X = np.array(X)
    X = X.reshape(300, -1)

    principalComponents = pca.fit_transform(X)

    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df['Inhibition']], axis=1)

    color = []
    label = []
    for i in range(len(finalDf)):
        if finalDf.loc[i, 'Inhibition'] <= 80:
            color.append('r')
            label.append('At or Below 80% Inhibition')
        else:
            color.append('b')
            label.append('Above 80% Inhibition')
    finalDf['color'] = color
    finalDf['label'] = label

    # PCA Plot.
    fig, ax = plt.subplots()
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA on Amine Fragments')
    for l in np.unique(label):
        ix = np.where(finalDf['label'] == l)
        ax.scatter(finalDf.loc[ix, 'principal component 1'], finalDf.loc[ix, 'principal component 2'],
                   c=finalDf.loc[ix, 'color'], label=l)
        ax.legend()
        plt.savefig('pca_2_comp.png')


    ### UMAP ###'
    mapper = umap.UMAP()
    principalComponents = mapper.fit_transform(X)

    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df['Inhibition']], axis=1)
    color = []
    label = []
    for i in range(len(finalDf)):
        if finalDf.loc[i, 'Inhibition'] <= 80:
            color.append('r')
            label.append('At or Below 80% Inhibition')
        else:
            color.append('b')
            label.append('Above 80% Inhibition')
    finalDf['color'] = color
    finalDf['label'] = label

    # UMAP Plot.
    fig, ax = plt.subplots()
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('UMAP on Amine Fragments')
    for l in np.unique(label):
        ix = np.where(finalDf['label'] == l)
        ax.scatter(finalDf.loc[ix, 'principal component 1'], finalDf.loc[ix, 'principal component 2'],
                   c=finalDf.loc[ix, 'color'], label=l)
        ax.legend()
        plt.savefig('umap.png')

if __name__ == '__main__':
    main()