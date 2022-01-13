'''
Finding the False Positives and False Negatives from
Gaussain Process Regression on Moonshot data.
'''

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF

# Defining the kernels
rbf_kernel = RBF()
matern_kernel = Matern()

rbf_gp = GaussianProcessRegressor(kernel=rbf_kernel)
matern_gp = GaussianProcessRegressor(kernel=matern_kernel)

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


'''
Leave one out for the train / test splits.
    Args:
        df (dataframe): The pre-split dataframe.

        index (int): The index you want to leave out
                     for testing.
'''


def leaveoneout_splits(df, index):
    test = df[index:index + 1]
    train = df.drop(index)

    return train, test


def main():
    # Importing the data.
    df = pd.read_csv('/Users/emmaking-smith/Moonshot/noisy_amides.csv', index_col=0)
    df = df.reset_index(drop=True)

    # Adding Inhibition (100 - Activity) column.
    df['Inhibition'] = 100 - df['Mean activity (%)']

    # Generating the amine fragments column.
    df = remove_acid(df)

    amine_rbf_preds = []
    amine_rbf_stds = []
    amine_matern_preds = []
    amine_matern_stds = []

    # Train / Test splits - Doing leave one out on entire dataframe.
    for i in range(len(df)):
        train_df, test_df = leaveoneout_splits(df, i)
        # Gaussian Process training on amines.
        rbf_gp.fit(train_amine_fingerprints, train_inhib)
        matern_gp.fit(train_amine_fingerprints, train_inhib)

        # Testing the gaussian processes on amines.
        amine_rbf_pred, amine_rbf_std = rbf_gp.predict(test_amine_fingerprints, return_std=True)
        amine_matern_pred, amine_matern_std = matern_gp.predict(test_amine_fingerprints, return_std=True)

        # Appending Results to Lists.
        amine_rbf_preds.append(amine_rbf_pred[0])
        amine_rbf_stds.append(amine_rbf_std[0])
        amine_matern_preds.append(amine_matern_pred[0])
        amine_matern_stds.append(amine_matern_std[0])

    df['amine_rbf_pred'] = amine_rbf_preds
    df['amine_matern_pred'] = amine_matern_preds

    rbf_label = []
    matern_label = []

    # Deteriming which points are false positives and false negatives based on a 20% deviation
    # between predicted and true.
    for i in range(len(df)):
        true = df.loc[i, 'Inhibition']
        rbf_pred = df.loc[i, 'amine_rbf_pred']
        matern_pred = df.loc[i, 'amine_matern_pred']
        print("true", true)
        print("rbf_pred", rbf_pred)

        # RBF false positives and false negatives.
        if rbf_pred > true + 20:
            rbf_label.append('FP')
        elif rbf_pred < true - 20:
            rbf_label.append('FN')
        else:
            rbf_label.append('TP/TN')

        # Matern false positives and false negatives.
        if matern_pred > true + 20:
            matern_label.append('FP')
        elif rbf_pred < true - 20:
            matern_label.append('FN')
        else:
            matern_label.append('TP/TN')

    df['rbf_label'] = rbf_label
    df['matern_label'] = matern_label

    rbf_outliers = df.loc[df['rbf_label'] != 'TP/TN', ['Molecule Name', 'SMILES', 'Mean activity (%)',
                                                       'Standard Deviation', 'Inhibition', 'rbf_label']]
    rbf_outliers.to_csv('/Users/emmaking-smith/Moonshot/rbf_outliers.csv')

    matern_outliers = df.loc[df['matern_label'] != 'TP/TN', ['Molecule Name', 'SMILES', 'Mean activity (%)',
                                                       'Standard Deviation', 'Inhibition', 'matern_label']]
    matern_outliers.to_csv('/Users/emmaking-smith/Moonshot/matern_outliers.csv')

    intersected_outliers = df.loc[(df['rbf_label'] != 'TP/TN') & (df['matern_label'] != 'TP/TN'),
                                  ['Molecule Name', 'SMILES', 'Mean activity (%)', 'Standard Deviation',
                                   'Inhibition', 'rbf_label', 'matern_label']]

    intersected_outliers.to_csv('/Users/emmaking-smith/Moonshot/intersected_outliers.csv')