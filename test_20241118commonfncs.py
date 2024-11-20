# Copyright 2024 Joon-Sang Park. All Rights Reserved.

import numpy as np

elecneg = {"C": 2.55, "N": 3.04, "O": 3.44, "S":2.58, "F":3.98, "P":2.19, "Cl":3.16, "Br":2.96, "B":2.04, "H":2.2, "Se":2.55, "I":2.05, "Xe":2.1, "Si": 1.9, "Xx":2.5} #Xx set to mean
t_six = [0, 1, 2, 3, 4, 5]

def one_hot_encoding(x, mset):
    x = x if (x in mset) else mset[-1]
    return list(map(lambda s: x == s, mset))

t_atom_dti = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']
t_atom = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'Se', 'I', 'Xe', 'Si', 'Xx']
t_degree = [0, 1, 2, 3, 4, 5]
t_totalnumhs = [0, 1, 2, 3, 4]
t_impval = [0, 1, 2, 3, 4, 5]

def getatomfeaturelen(newfeat = False):
    base = len(t_atom_dti) if (not newfeat) else len(list(elecneg)) + 1 #elect neg
    return len(t_degree) + len(t_totalnumhs) + len(t_impval) + base + 1 #isaromatic

def atom_feature(m, atom_i, include_implicitvalence=False, include_elecneg=False):
    atom = m.GetAtomWithIdx(atom_i)
    atsym = atom.GetSymbol()
    
    features = one_hot_encoding(atsym, t_atom_dti) + \
               one_hot_encoding(atom.GetDegree(), t_degree) + \
               one_hot_encoding(atom.GetTotalNumHs(), t_totalnumhs)
    
    if include_implicitvalence:
        features += one_hot_encoding(atom.GetImplicitValence(), t_impval)
    
    if include_elecneg:
        elecneg_value = elecneg.get(atsym, elecneg["Xx"])
        features.append(elecneg_value)
    
    features.append(atom.GetIsAromatic())
    
    return np.array(features)

    
    """
    return np.array(one_hot_encoding(atsym, t_atom_dti) +
                    one_hot_encoding(atom.GetDegree(), t_degree) +
                    one_hot_encoding(atom.GetTotalNumHs(), t_totalnumhs) +
                    one_hot_encoding(atom.GetImplicitValence(), t_impval) +
                    [atom.GetIsAromatic()])
                    # (10, 6, 5, 6, 1) --> total 28
    """
def featurize(m, atom_i, include_implicitvalence=False, include_elecneg=False):
    atom = m.GetAtomWithIdx(atom_i)
    atsym = atom.GetSymbol()

    features= []
    features.extend(one_hot_encoding(atsym, t_atom))
    features.extend(one_hot_encoding(atom.GetDegree(), t_degree))
    features.extend(one_hot_encoding(atom.GetTotalNumHs(), t_totalnumhs))
    # implicitvalence 피처
    if include_implicitvalence:
        features.extend(one_hot_encoding(atom.GetImplicitValence(), t_impval))
    # 방향족 여부
    features.append(atom.GetIsAromatic())
    # 전기 음성도 피처
    if include_elecneg:
        features.append(elecneg[atsym if (atsym in t_atom) else t_atom[-1]])
    
    return np.array(features)
    """
    return np.array(one_hot_encoding(atsym, t_atom) +
                    one_hot_encoding(atom.GetDegree(), t_degree) +
                    one_hot_encoding(atom.GetTotalNumHs(), t_totalnumhs) +
                    one_hot_encoding(atom.GetImplicitValence(), t_impval) +
                    [atom.GetIsAromatic()] +
                    [elecneg[atsym if (atsym in t_atom) else t_atom[-1]]])
                    # (10+5, 6, 5, 6, 1) + 1 --> total 28+6=34
    """

def getatomfeaturelen_f():
    return len(list(elecneg)) + len(t_six) + len(t_six) + 1 + 1

def featurize_f(m, atom_i, include_implicitvalence=False, include_elecneg=False):
    atom = m.GetAtomWithIdx(atom_i)
    atsym = atom.GetSymbol()
    atomtype = list(elecneg)

    features = []
    # 기본 피처
    features.extend(one_hot_encoding(atsym, atomtype))
    features.extend(one_hot_encoding(atom.GetDegree(), t_six))

    # implicitvalence 피처
    if include_implicitvalence:
        features.extend(one_hot_encoding(atom.GetTotalNumHs() + atom.GetImplicitValence(), t_six))
    else:
        features.extend(one_hot_encoding(atom.GetTotalNumHs(), t_six))
    
    # 방향족 여부
    features.append(atom.GetIsAromatic())
    
    # 전기 음성도 피처
    if include_elecneg:
        features.append(elecneg[atsym if (atsym in atomtype) else atomtype[-1]])
    
    return np.array(features)    

    """
    return np.array(one_hot_encoding(atsym, atomtype) +
                    one_hot_encoding(atom.GetDegree(), t_six) +
                    one_hot_encoding(atom.GetTotalNumHs()+atom.GetImplicitValence(), t_six) +
                    [atom.GetIsAromatic()] +
                    [elecneg[atsym if (atsym in atomtype) else atomtype[-1]]])
                    # 15 + 6 + 6 + 1 + 1 = 29
    """
def get_atom_feature_f(m, dist_ligand = 0, include_implicitvalence=False, include_elecneg=False):
    n = m.GetNumAtoms()
    atom_feature_len = getatomfeaturelen_f()
    if not include_implicitvalence:
        atom_feature_len -= len(t_six)  # implicitvalence를 포함하지 않을 경우

    H = np.empty((n, atom_feature_len))
    for i in range(n):
        H[i] = featurize_f(m, i, include_implicitvalence, include_elecneg)
    if 0 == dist_ligand:
        return H
    if dist_ligand == 1:
        H = np.concatenate([H, np.zeros((n,atom_feature_len))], 1)
    else:
        H = np.concatenate([np.zeros((n,atom_feature_len)), H], 1)
    return H

def get_atom_feature(m, is_ligand=True, include_implicitvalence=False, include_elecneg=False):
    n = m.GetNumAtoms()
    atom_feature_len = len(one_hot_encoding('', t_atom)) + len(t_degree) + len(t_totalnumhs) + 1  # 기본 피처
    if include_implicitvalence:
        atom_feature_len += len(t_impval)
    if include_elecneg:
        atom_feature_len += 1

    H = np.empty((n, atom_feature_len))
    for i in range(n):
        H[i] = featurize(m, i, include_implicitvalence, include_elecneg)
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,atom_feature_len))], 1)
    else:
        H = np.concatenate([np.zeros((n,atom_feature_len)), H], 1)
    return H