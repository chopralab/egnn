from collections import defaultdict
import os
import pickle
import io
import sys

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




if __name__ == "__main__":

    DATASET, radius = sys.argv[1:]
    radius = int(radius)

    with io.open('input/data.txt', mode='r') as f:
        data_list = f.read().strip().split('\n')

    """Exclude data contains "." in its smiles."""

    N = len(data_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    Smiles, Molecules, adjacencies, properties = '', [], [], []
    
    score = []
    
       
    for no, data in enumerate(data_list):

        print('/'.join(map(str, [no+1, N])))

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

        for i in data.strip().split('\t')[1:-1]:
            score.append(np.float32(i))        
        ##scores            
        
    separate_energies = [[] for _ in range(int((len(score))/int(N)))]  


    for _ in separate_energies:
        for i in range(96):
            _.append(np.array([np.float32(score[i::96])]))
            

    
    normalized_separate_energies = [[] for _ in range(int((len(score))/int(N)))]
    for _ in normalized_separate_energies:
        for l in separate_energies[0]:
            for i in l:
                _.append(normalize_score(i))


    normalized_separate_energies = normalized_separate_energies[0]
#    print(normalized_separate_energies[0][0])
#    print(normalized_separate_energies[1][0])
#    print(normalized_separate_energies[2][0])

    concatenated_normalized_separate_energies = np.concatenate((normalized_separate_energies[:]), axis=0)

    
    normalized_separate_molecules = []
    for i in range(N):
        normalized_separate_molecules.append(concatenated_normalized_separate_energies[i::N])
        
    
    docking_scores = normalized_separate_molecules  
#    print(docking_scores)
    
    dir_train = ('train/radius' + str(radius) + '/')
    os.makedirs(dir_train, exist_ok=True)

    with open(dir_train + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_train + 'Molecules', Molecules)
    np.save(dir_train + 'adjacencies', adjacencies)
    np.save(dir_train + 'properties', properties)
    np.save(dir_train + 'docking_scores', docking_scores)    


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

        for i in test.strip().split('\t')[1:]:
            score.append(np.float32(i))
        ##scores            
        
    separate_energies = [[] for _ in range(int((len(score))/int(N)))]  


    for _ in separate_energies:
        for i in range(96):
            _.append(np.array([np.float32(score[i::96])]))
            

    
    normalized_separate_energies = [[] for _ in range(int((len(score))/int(N)))]
    for _ in normalized_separate_energies:
        for l in separate_energies[0]:
            for i in l:
                _.append(normalize_score(i))


    normalized_separate_energies = normalized_separate_energies[0]
#    print(normalized_separate_energies[0][0])
#    print(normalized_separate_energies[1][0])
#    print(normalized_separate_energies[2][0])

    concatenated_normalized_separate_energies = np.concatenate((normalized_separate_energies[:]), axis=0)

    
    normalized_separate_molecules = []
    for i in range(N):
        normalized_separate_molecules.append(concatenated_normalized_separate_energies[i::N])
    
    docking_scores = normalized_separate_molecules  
    print(docking_scores)        
        

    dir_test = ('test/radius' + str(radius) + '/')
    os.makedirs(dir_test, exist_ok=True)

    with open(dir_test + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_test + 'Molecules', Molecules)
    np.save(dir_test + 'adjacencies', adjacencies)
    np.save(dir_test + 'docking_scores', docking_scores) 

    dump_dictionary(fingerprint_dict, dir_train + 'fingerprint_dict.pickle')
    dump_dictionary(fingerprint_dict, dir_test  + 'fingerprint_dict.pickle')
