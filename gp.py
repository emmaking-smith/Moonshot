'''
A Gaussian Process Regressor for predicting inhibition activity of various amide scaffolds.

THERE ARE 300 UNIQUE SMILES AND 300 UNIQUE CANONICAL SMILES

Leave one out.
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
Leave one out for the train / test splits.
    Args:
        df (dataframe): The pre-split dataframe.
        
        index (int): The index you want to leave out
                     for testing.
'''
def leaveoneout_splits(df, index):
    test = df[index:index+1]
    train = df.drop(index)

    return train, test

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

    rbf_preds = []
    rbf_stds = []
    matern_preds = []
    matern_stds = []

    amine_rbf_preds = []
    amine_rbf_stds = []
    amine_matern_preds = []
    amine_matern_stds = []

    # Train / Test splits - Doing leave one out on entire dataframe.
    for i in range(len(df)):
        train_df, test_df = leaveoneout_splits(df, i)

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

        # Gaussian Process training on whole molecules.
        rbf_gp.fit(train_fingerprints, train_inhib)
        matern_gp.fit(train_fingerprints, train_inhib)

        # Testing the gaussian processes on whole molecules.
        rbf_pred, rbf_std = rbf_gp.predict(test_fingerprints, return_std=True)
        print("rbf_pred", rbf_pred[0])
        print("rbf_std", rbf_std[0])
        matern_pred, matern_std = matern_gp.predict(test_fingerprints, return_std=True)
        print("matern_pred", matern_pred[0])
        print("matern_std", matern_std[0])
        # Appending Results to Lists.
        rbf_preds.append(rbf_pred[0])
        rbf_stds.append(rbf_std[0])
        matern_preds.append(matern_pred[0])
        matern_stds.append(matern_std[0])

        # Gaussian Process training on amines.
        rbf_gp.fit(train_amine_fingerprints, train_inhib)
        matern_gp.fit(train_amine_fingerprints, train_inhib)

        # Testing the gaussian processes on amines.
        amine_rbf_pred, amine_rbf_std = rbf_gp.predict(test_amine_fingerprints, return_std=True)
        amine_matern_pred, amine_matern_std = matern_gp.predict(test_amine_fingerprints, return_std=True)

        # Appending Results to Lists.
        amine_rbf_preds.append(amine_rbf_pred[0])
        amine_rbf_stds.append(amine_rbf_std[0])
        print("amine_rbf_pred", amine_rbf_pred[0])
        amine_matern_preds.append(amine_matern_pred[0])
        print("amine_matern_pred", amine_matern_pred[0])
        amine_matern_stds.append(amine_matern_std[0])

    # RBF Plots (whole molecules).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('RBF Gaussian Process Regression (Leave One Out)')
    plt.errorbar(np.array(df['Inhibition']), np.array(rbf_preds), yerr=2*1.96*np.array(rbf_stds), fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i,xy in enumerate(zip(df['Inhibition'], rbf_preds)):
        ax.annotate(xy=xy, text=str(i))
    plt.savefig('loo_rbf.png')

    # Matern Plots (whole molecules).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('Matern Gaussian Process Regression (Leave One Out)')
    plt.errorbar(np.array(df['Inhibition']), np.array(matern_preds), yerr=2*1.96*np.array(matern_stds), fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i,xy in enumerate(zip(df['Inhibition'], matern_preds)):
        ax.annotate(xy=xy, text=str(i))
    plt.savefig('loo_matern.png')
    print("")
    # RBF Plots (amines).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('RBF Gaussian Process Regression (Leave One Out, Amine Fragments)')
    plt.errorbar(np.array(df['Inhibition']), np.array(amine_rbf_preds), yerr=2*1.96*np.array(amine_rbf_stds), fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i,xy in enumerate(zip(df['Inhibition'], amine_rbf_preds)):
        ax.annotate(xy=xy, text=str(i))
    plt.savefig('loo_amine_rbf.png')

    # Matern Plots (amines).
    fig, ax = plt.subplots()
    # y = x line.
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='b')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('Matern Gaussian Process Regression (Leave One Out, Amine Fragments)')
    plt.errorbar(np.array(df['Inhibition']), np.array(amine_matern_preds), yerr=2*1.96*np.array(amine_matern_stds), fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    for i,xy in enumerate(zip(df['Inhibition'], amine_matern_preds)):
        ax.annotate(xy=xy, text=str(i))
    plt.savefig('loo_amine_matern.png')

if __name__ == '__main__':
    main()