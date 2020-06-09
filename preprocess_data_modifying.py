from collections import defaultdict
import pandas as pd
import pickle
import sys
import io
import os

import numpy as np

from rdkit import Chem


def create_atoms(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')  # Consider aromaticity.
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def create_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (or fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update vertex IDs considering neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update edge IDs considering nodes on both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def normalize_score(dataset):
    norm_list = list()
    min_value = min(dataset)
    max_value = max(dataset)
    
    for value in dataset:
            tmp = (value - min_value) / (max_value - min_value)
            norm_list.append(tmp)
    return norm_list    
#    return (((float(i)-min(lst))/(max(lst)-min(lst))) for i in lst )

   


if __name__ == "__main__":

    DATASET, radius = sys.argv[1:]
    radius = int(radius)

    with io.open('input/data.txt', mode="r") as f:
        data_list = f.read().strip().split('\n')
    

    """Exclude data contains "." in its smiles."""
    data_list = list(filter(lambda x:
                     '.' not in x.strip().split()[0], data_list))
#    print(data_list)
        
#    score1 = []
#        for row in data_list:
#            score1.append(data.strip().split('\t')[1])
#        print(score1)
        
    N = len(data_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    Smiles, Molecules, adjacencies, properties = '', [], [], []

    score = []
   

  
     
    for no, data in enumerate(data_list):
        
#        score = [[] for _ in range(len(data.strip().split('\t')[1:-1]))]
        
        print('/'.join(map(str, [no+1, N])))
        
#        score = [[] for l in range(len(data.strip().split('\t')[1:-1]))] 
        
        smiles, property = data.strip().split('\t')[0], data.strip().split('\t')[-1]
        Smiles += smiles + '\n'

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = create_fingerprints(atoms, i_jbond_dict, radius)
        Molecules.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)
        
        property = np.array([float(property)])
        properties.append(property)

        #scores_auto
        scores = data.strip().split('\t')[1:-1]
        score.append(scores)
    
    score = np.array(score, dtype=np.float32)

#    print(properties)
#    print(score)

    df=pd.DataFrame(score)

    
    docking_scores = []
    
    for columnName, columnData in df.iteritems():
        docking_scores.append(normalize_score(columnData))
        df[columnName] = normalize_score(columnData)
#    print(df)
#    print(docking_scores)

    normalized_scores = df.to_numpy().tolist()
    normalized_scores = np.array(normalized_scores, dtype=np.float32)
    print(normalized_scores)

    
#    
#    docking_scores = normalized_separate_molecules  
##    print(docking_scores[0])
##    print(docking_scores)
#    
#    
#
    dir_train = ('train/radius'+str(radius)+'/')
    os.makedirs(dir_train, exist_ok=True)
#
    with open(dir_train + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_train + 'Molecules', Molecules)
    np.save(dir_train + 'adjacencies', adjacencies)
    np.save(dir_train + 'properties', properties)
    np.save(dir_train + 'docking_scores', normalized_scores)

    with io.open('input/test.txt', mode='r') as f:
        test_list = f.read().strip().split('\n')


    N = len(test_list)

    Smiles, Molecules, adjacencies = '', [], []
    score = []

    for no, test in enumerate(test_list):

        print('/'.join(map(str, [no+1, N])))

        smiles = test.strip().split('\t')[0]
        Smiles += smiles + '\n'

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = create_fingerprints(atoms, i_jbond_dict, radius)
        Molecules.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)


        scores = test.strip().split('\t')[1:]
        score.append(scores)
    
    score = np.array(score, dtype=np.float32)

#    print(properties)
#    print(score)

    df=pd.DataFrame(score)

    
    docking_scores = []
    
    for columnName, columnData in df.iteritems():
        docking_scores.append(normalize_score(columnData))
        df[columnName] = normalize_score(columnData)
#    print(df)
#    print(docking_scores)

    normalized_scores = df.to_numpy().tolist()
    normalized_scores = np.array(normalized_scores, dtype=np.float32)
    print(normalized_scores)
#    print(docking_scores)        
        

    dir_test = ('synthetic_test/radius' + str(radius) + '/')
    os.makedirs(dir_test, exist_ok=True)

    with open(dir_test + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_test + 'Molecules', Molecules)
    np.save(dir_test + 'adjacencies', adjacencies)
    np.save(dir_test + 'docking_scores', normalized_scores)


    dump_dictionary(fingerprint_dict, dir_train + 'fingerprint_dict.pickle')
    dump_dictionary(fingerprint_dict, dir_test  + 'fingerprint_dict.pickle')


    
 
    
