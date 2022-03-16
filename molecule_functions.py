import math

def GetXYZ(file):
    """
    Returns 4 lists: a list of atoms, x coordinates, y coordinates and z coordinates.
    """
    atom=[]
    X=[]
    Y=[]
    Z=[]

    g=open(file,'r+', errors='ignore')

    lines=g.readlines()

    g.close()

    for line in lines[2:]:
        atom.append(line.split()[0])
        X.append(float(line.split()[1]))
        Y.append(float(line.split()[2]))
        Z.append(float(line.split()[3]))
    
    return atom, X, Y, Z

def GetSystemSize(file):
    f=open(file,'r+')
    lines=f.readlines()
    f.close()
    return len(lines[2:])

def GetDists(n, n_atoms_on_mol, file):
    """
    Creates a list of distances for a given atom "n"(parameter of the function) from all the cluster atoms
    """
    atom=[]
    X=[]
    Y=[]
    Z=[]
    
    atom, X, Y, Z = GetXYZ(file)
    
    dists=[]
    SystemSize=GetSystemSize(file)
    for num in range(n_atoms_on_mol,SystemSize):
        d=math.sqrt((X[n]-X[num])**2 + (Y[n]-Y[num])**2 + (Z[n]-Z[num])**2)
        dists.append(d)
    
    return atom[0:n_atoms_on_mol], atom[n_atoms_on_mol:], dists

def MolMinDists(file, n_atoms_on_mol):
    """
    Returns the minimal bond distance for each atom of the molecule from the cluster as well as
    which are the atoms involved in the bond
    """
    minimal_distances=[]
    atoms=[]
    for n in range(n_atoms_on_mol):
        mol_atoms, cluster_atoms, n_dists=GetDists(n,n_atoms_on_mol, file)
        min_n_dists=min(n_dists)
        minimal_distances.append(min_n_dists)
        
        atom_i_min_dist=n_dists.index(min_n_dists)
        closer_cluster_atom=cluster_atoms[atom_i_min_dist]
        atoms.append([closer_cluster_atom, mol_atoms[n]])
    
    MinDist_mol_cluter=min(minimal_distances)
    bond_mol_cluster=atoms[minimal_distances.index(MinDist_mol_cluter)]
    
    return MinDist_mol_cluter, bond_mol_cluster, minimal_distances, atoms

def BondDist(A, B, file):
    X=[]
    Y=[]
    Z=[]
    _, X, Y, Z= GetXYZ(file)
    distAB=math.sqrt((X[A]-X[B])**2 + (Y[A]-Y[B])**2 + (Z[A]-Z[B])**2)
    return distAB

def BondAngle(A, B, C, file):
    X=[]
    Y=[]
    Z=[]
    _,X, Y, Z= GetXYZ(file)
    
    angleABC=math.acos((BondDist(A,B, file)**2 + BondDist(B,C, file)**2 - BondDist(A,C, file)**2)/(2*BondDist(A,B, file)*BondDist(B,C, file)))
    return angleABC

def GetEnergy(filename):
    filename = filename.split('/')[-1]
    energy = float('0'+filename.split('_')[-2])
    return energy

def OrderAtomsByDist(d1, a1, d2, a2):
    bond_lengths=sorted([d1, d2])
    
    if d1 > d2:
        close = a2
        far = a1
    else:
        close = a1
        far = a2

    return bond_lengths, close, far

def closest_cluster_atom(mol_atom, bond_list):
    return bond_list[mol_atom][0]