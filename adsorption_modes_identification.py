import os
import abc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob               import glob
from molecule_functions import *

from importlib.resources       import path
from sklearn.decomposition     import PCA
from sklearn.preprocessing     import StandardScaler
from yellowbrick.cluster.elbow import KElbowVisualizer
from sklearn.cluster           import KMeans

class AdsorptionModesIdentifier():
    def __init__(self, 
                data_collector,
                data_transformer,
                data_classifier,
                PCA_image: bool = False) -> None:
        self.PCA_image = PCA_image
        self.collector = data_collector
        self.transformer = data_transformer
        self.classifier = data_classifier

    def identify_modes(self) -> None:
        original_data = self.collector.obtain_data()
        data = self.transformer.transform_data(original_data)
        labels = self.classifier.modes_classifier(data)

        if self.PCA_image:
            self._create_PCA_image(data, self.classifier.k)
        
        original_data['labels'] = labels.values

        original_data.to_csv('./clustered_data.csv')

        return None

    def _create_PCA_image(self, X, k) -> None:
        pca = PCA(n_components = min(X.shape))

        principal_components = pca.fit_transform(X)

        # plot explained variance
        features = range(pca.n_components_)
        
        plt.figure()
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.savefig('PCA_components_2.png')
        
        kmeans = KMeans(n_clusters= self.classifier.k , random_state=42, n_init=300)
        kmeans.fit(X)
        
        plt.figure()
        df_pca = pd.DataFrame(principal_components)
        df_pca['labels'] = kmeans.labels_
        sns.scatterplot(x=0, y=1, data=df_pca, hue='labels', palette='rainbow')

        x_label = f"PCA 1 ({pca.explained_variance_ratio_[0]:.2f})"
        y_label = f"PCA 2 ({pca.explained_variance_ratio_[1]:.2f})"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig('PCA_distribution_2.png')

        return None

class DataCollector(abc.ABC):
    def __init__(self, path = os.getcwd()) -> None:
        if os.path.exists(path):
            self.path = path
        else:
            raise IOError('path for structures does not exist')
    
    def _get_configuration_number(self, filename) -> int:
        filename = filename.split('/')[-1]
        conf_num = filename.split('_')[-1].split('.')[0]
        
        return conf_num

    @abc.abstractmethod
    def _get_molecule_data():
        pass

    @abc.abstractmethod
    def _get_columns_names():
        pass

    def obtain_data(self) -> pd.DataFrame:     
        df_cols = self._get_columns_names()
        df_cols.insert(0, 'configuration_number')
        
        df_structures_data = pd.DataFrame(columns=df_cols)

        for file in glob(self.path+'/*.xyz'):
            conf_num = self._get_configuration_number(file)
            structure_data = self._get_molecule_data(file)
            structure_data['configuration_number'] = int(conf_num)


            df_structures_data = pd.concat([df_structures_data, structure_data], axis=0)

        df_structures_data = df_structures_data.set_index('configuration_number')
        
        return df_structures_data

class DataTransformer():
    def __init__(self, embedding=PCA(), scaler=StandardScaler()):
        self.embedding = embedding
        self.scaler = scaler

    def _categorical_columns(self, data: pd.DataFrame) -> list:
        return data.select_dtypes(include=['object']).columns

    def _numerical_columns(self, data: pd.DataFrame) -> list:
        return data.select_dtypes(exclude=['object']).columns
          
    def _apply_scaling(self, data:pd.DataFrame) -> pd.DataFrame:
        scaled_df = pd.DataFrame(self.scaler.fit_transform(data), columns = data.columns)
        return scaled_df

    def _apply_embedding(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.embedding.fit_transform(data))

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        to_drop_columns = ['configuration_number','energy']
        data = data.drop([col for col in to_drop_columns if col in data.columns], axis=1, errors='ignore')

        categorical_data = data[self._categorical_columns(data)]
        numerical_data   = data[self._numerical_columns(data)]
        
        if not categorical_data.empty:
            categorical_data = pd.get_dummies(categorical_data).reset_index()

        numerical_data = self._apply_scaling(numerical_data)

        data = pd.concat([numerical_data, categorical_data], axis=1)

        data = self._apply_embedding(data)

        return data

class DataClassifier():
    def __init__(self, k:int = None, metric_k: str = 'silhouette') -> None:
        self.k = k
        if self.k == None:
            self.metric_k = metric_k #could be distortion, silhouette or calinski_harabasz
    
    def _choose_K(self, scaled_data: pd.DataFrame, metric: str) -> int:
        visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2,10), metric = metric).fit(scaled_data)
        k = visualizer.elbow_value_ 
        visualizer.show('silhouette_score.png')

        return k

    def modes_classifier(self, scaled_data: pd.DataFrame) -> pd.DataFrame:
        if self.k == None:
            self.k = self._choose_K(scaled_data, self.metric_k)

        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=300)
        kmeans.fit(scaled_data)
        
        scaled_data['labels'] = kmeans.labels_
        
        return scaled_data['labels']


class MoleculesCollector(DataCollector):
    def __init__(self, path) -> None:
        super().__init__(path=path)
        setup_info = self._setup()
        self.mol_info = setup_info['molecule_info']

    def _setup(self):
        with open('setup.in', 'r') as f:
            setup = {}
            for line in f:
                if line.strip():
                    (key, val) = line.replace(' ','').strip().split(':')
                    setup[key] = val
        
        return setup

    def _molecule_atoms(self):
        intervals = self.mol_info.split(',')
        indexes = []
        
        for interval in intervals:
            if '-' in interval:
                lower_limit, upper_limit = interval.split('-')
                # to_append = [num for num in range(int(lower_limit), int(upper_limit))]
                indexes += [num-1 for num in range(int(lower_limit), int(upper_limit)+1)]
            elif int(interval):
                indexes.append(int(interval)-1)

        return set(indexes)

    def _get_lowest_bonds(self, filename: str):             
        atoms, X, Y, Z = GetXYZ(filename)
        system_size = GetSystemSize(filename)
        molecule_atoms_indexes = self._molecule_atoms()

        substrate_indexes = set(range(system_size)) - molecule_atoms_indexes

        atomic_distances = []

        for atom in molecule_atoms_indexes:
            molecule_atom_distances = []            
            
            for specie in substrate_indexes:
                d = BondDist(atom, specie, filename)
                molecule_atom_distances.append((d, specie ,atoms[specie]))

            atomic_distances.append(min(molecule_atom_distances)+(atom, atoms[atom]))


        return [x[::-1] for x in atomic_distances]              

    def _closest_molecule_atom(self, ordered_atoms, filename):
        atoms, X, Y, Z = GetXYZ(filename)

        atomic_distances = []

        for atom in ordered_atoms:
            molecule_distances = []

            for other in [elem for elem in ordered_atoms if elem != atom]:
                d = BondDist(atom, other, filename)
                molecule_distances.append((d, other, atoms[other]))

            atomic_distances.append(min(molecule_distances)+(atom,atoms[atom]))

        return [x[::-1] for x in atomic_distances] 


    def _get_intramolecular_angles(self, ordered_atoms, bonds_data, filename):                
        atoms, _, _, _ = GetXYZ(filename)
        
        closest_atom_in_substrate ={element[1]:element[3] for element in bonds_data}
        closest_atom_in_molecule = self._closest_molecule_atom(ordered_atoms, filename)
        closest_atom_in_molecule = {element[1]:element[3] for element in closest_atom_in_molecule}

        intramolecular_angles = []

        for atom in ordered_atoms:
            angle = BondAngle(closest_atom_in_molecule[atom],atom,closest_atom_in_substrate[atom],filename)
            bond_dist_sum = BondDist(closest_atom_in_molecule[atom],atom, filename) + BondDist(atom,closest_atom_in_substrate[atom], filename)
            intramolecular_angles.append((atom, atoms[atom], closest_atom_in_substrate[atom], atoms[closest_atom_in_substrate[atom]], closest_atom_in_molecule[atom], atoms[closest_atom_in_molecule[atom]], angle, bond_dist_sum))
        
        return intramolecular_angles
            
    def _get_molecule_data(self, filename: str):
        bonds_data = self._get_lowest_bonds(filename)
        bonds_data = sorted(bonds_data, key=lambda tup: tup[4])

        distances = [float(element[4]) for element in bonds_data]
        bonds = [element[0]+'-'+element[2] for element in bonds_data]
        ordered_atoms = tuple(element[1] for element in bonds_data)

        angles = self._get_intramolecular_angles(ordered_atoms, bonds_data, filename)
        angle_values = [float(element[6]) for element in angles]
        angle_strings = [element[3]+'-'+element[1]+'-'+element[5] for element in angles]
        bond_dist_sums = [float(element[7]) for element in angles]
        
        df_conf = pd.DataFrame(data=[[*distances, *bonds, *angle_values, *angle_strings, *bond_dist_sums]])

        return df_conf

    def _get_columns_names(self):
        names = len(self._molecule_atoms())
        cols_names = [num for num in range(names*5)]

        return cols_names

if __name__ == '__main__':
    mol_path=sys.argv[1]

    identifier = AdsorptionModesIdentifier( data_collector   = MoleculesCollector(path = mol_path),
                                            data_transformer = DataTransformer(),
                                            data_classifier  = DataClassifier(),
                                            PCA_image        = True )

    identifier.identify_modes()