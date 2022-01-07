'''
A Gaussian Process Regressor for predicting inhibition activity of various amide scaffolds.
'''
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF

# Defining the kernels
rbf_kernel = RBF()
matern_kernel = Matern()

rbf_gp = GaussianProcessRegressor(kernel=rbf_kernel)
matern_gp = GaussianProcessRegressor(kernel=matern_kernel)

'''
Splitting the dataframe to training and testing datasets. Leave
out user defined percentage of molecules.
    Args:
        df (dataframe): The pre-split dataframe.
        
        test_ratio (float): The percentage of molecules
                            to be left out for testing.
'''
def train_test_split(df, test_ratio=0.05):
    random.seed(1)
    smiles = df['SMILES']
    # Finding the smiles that will be used for the test dataset.
    test_smiles = random.sample(smiles.to_list(), int(np.ceil(test_ratio*len(df))))
    test_df = pd.DataFrame()

    # Splitting the dataframe into test and non-test (train).
    for smile in test_smiles:
        test_row = df[df['SMILES'] == smile]
        test_df = pd.concat([test_df, test_row])
        df = df[df['SMILES'] != smile]

    return df, test_df

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

    # Adding Inhibition (100 - Activity) column.
    df['Inhibition'] = 100 - df['Mean activity (%)']

    # Generating the amine fragments column.
    df = remove_acid(df)

    # Train / Test splits.
    train_df, test_df = train_test_split(df, test_ratio=0.05)

    # Converting the whole molecule to fingerprints.
    train_smiles = train_df['SMILES'].to_list()
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    train_fingerprints = [Chem.RDKFingerprint(m) for m in train_mols]
    train_fingerprints = np.ravel(train_fingerprints).reshape(len(train_smiles), -1)

    test_smiles = test_df['SMILES'].to_list()
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    test_fingerprints = [Chem.RDKFingerprint(m) for m in test_mols]
    test_fingerprints = np.ravel(test_fingerprints).reshape(len(test_smiles), -1)

    # Converting the amines to fingerprints.
    train_amine_mols = train_df['Amine'].to_list()
    train_amine_fingerprints = [Chem.RDKFingerprint(m) for m in train_amine_mols]
    train_amine_fingerprints = np.ravel(train_amine_fingerprints).reshape(len(train_amine_mols), -1)

    test_amine_mols = test_df['Amine'].to_list()
    test_amine_fingerprints = [Chem.RDKFingerprint(m) for m in test_amine_mols]
    test_amine_fingerprints = np.ravel(test_amine_fingerprints).reshape(len(test_amine_mols), -1)

    # Retrieving the labels for the train / test sets.
    train_inhib = train_df['Inhibition']
    test_inhib = test_df['Inhibition']

    # Gaussian Process training on whole molecules.
    rbf_gp.fit(train_fingerprints, train_inhib)
    matern_gp.fit(train_fingerprints, train_inhib)

    # Testing the gaussian processes on whole molecules.
    rbf_preds, rbf_std = rbf_gp.predict(test_fingerprints, return_std=True)
    matern_preds, matern_std = matern_gp.predict(test_fingerprints, return_std=True)
    rbf_mse = ((test_inhib - rbf_preds)**2).mean()
    print("rbf_mse", rbf_mse)
    matern_mse = ((test_inhib - matern_preds)**2).mean()
    print("matern_mse", matern_mse)

    # Gaussian Process training on amines.
    rbf_gp.fit(train_amine_fingerprints, train_inhib)
    matern_gp.fit(train_amine_fingerprints, train_inhib)

    # Testing the gaussian processes on amines.
    amine_rbf_preds, amine_rbf_std = rbf_gp.predict(test_amine_fingerprints, return_std=True)
    amine_matern_preds, amine_matern_std = matern_gp.predict(test_amine_fingerprints, return_std=True)
    amine_rbf_mse = ((test_inhib - amine_rbf_preds) ** 2).mean()
    print("amine_rbf_mse", amine_rbf_mse)
    matern_mse = ((test_inhib - amine_matern_preds) ** 2).mean()
    print("amine_matern_mse", matern_mse)

    # RBF Plots (whole molecules).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('RBF Gaussian Process Regression')
    plt.errorbar(test_inhib, rbf_preds, yerr=1.96*rbf_std, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i,xy in enumerate(zip(test_inhib, rbf_preds)):
        ax.annotate(xy=xy, text=test_df.index[i])
    plt.savefig('rbf.png')

    # Matern Plots (whole molecules).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('Matern Gaussian Process Regression')
    plt.errorbar(test_inhib, matern_preds, yerr=1.96 * matern_std, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i,xy in enumerate(zip(test_inhib, matern_preds)):
        ax.annotate(xy=xy, text=test_df.index[i])
    plt.savefig('matern.png')

    # RBF Plots (amines).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('RBF Gaussian Process Regression (Amine Fragments)')
    plt.errorbar(test_inhib, amine_rbf_preds, yerr=1.96 * amine_rbf_std, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i,xy in enumerate(zip(test_inhib, amine_rbf_preds)):
        ax.annotate(xy=xy, text=test_df.index[i])
    plt.savefig('rbf_amine.png')

    # Matern Plots (amines).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('Matern Gaussian Process Regression (Amine Fragments)')
    plt.errorbar(test_inhib, amine_matern_preds, yerr=1.96 * amine_matern_std, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i, xy in enumerate(zip(test_inhib, amine_matern_preds)):
        ax.annotate(xy=xy, text=test_df.index[i])
    plt.savefig('matern_amine.png')

if __name__ == '__main__':
    main()