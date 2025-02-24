import numpy as np
from scipy.special import i0  # Fonction de Bessel modifiée d'ordre 0
from sklearn.mixture import GaussianMixture

class VonMisesMixture:
    def __init__(self, n_components=4):
        """Initialise un mélange de distributions de von Mises."""
        self.n_components = n_components
        self.weights = np.ones(n_components) / n_components
        self.means = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        self.concentrations = np.ones(n_components)

    def von_mises_density(self, theta, mu, kappa):
        """Calcule la densité de la loi von Mises."""
        return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))

    def fit(self, directions):
        """Met à jour les paramètres en utilisant un algorithme E-M adapté aux directions circulaires."""
        # Étape E : Calcul des responsabilités
        responsibilities = np.zeros((len(directions), self.n_components))
        for i in range(self.n_components):
            responsibilities[:, i] = self.weights[i] * self.von_mises_density(directions, self.means[i], self.concentrations[i])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # Étape M : Mise à jour des paramètres
        for i in range(self.n_components):
            weight_sum = responsibilities[:, i].sum()
            self.weights[i] = weight_sum / len(directions)
            self.means[i] = np.arctan2(
                np.sum(responsibilities[:, i] * np.sin(directions)),
                np.sum(responsibilities[:, i] * np.cos(directions))
            )
            resultant_length = np.sqrt(
                np.sum(responsibilities[:, i] * np.cos(directions))**2 +
                np.sum(responsibilities[:, i] * np.sin(directions))**2
            ) / weight_sum
            self.concentrations[i] = -np.log(resultant_length) / (1 - resultant_length)

    def predict_density(self, theta):
        """Calcule la densité de probabilité pour un angle donné."""
        density = sum(
            self.weights[i] * self.von_mises_density(theta, self.means[i], self.concentrations[i])
            for i in range(self.n_components)
        )
        return density


def compute_directional_probabilities(block_vectors):
    """Calcule la probabilité directionnelle pour chaque bloc avec un mélange de von Mises."""
    block_probabilities = {}

    for block, vectors in block_vectors.items():
        directions = np.array([v[2] for v in vectors])  # Extrait les angles des vecteurs de mouvement
        von_mises_model = VonMisesMixture(n_components=4)
        von_mises_model.fit(directions)
        block_probabilities[block] = von_mises_model

    return block_probabilities


'''def compute_magnitude_model(block_vectors):
    block_magnitudes = {}
    for block, vectors in block_vectors.items():
        magnitudes = np.array([v[3] for v in vectors])
        magnitudes = magnitudes.reshape(-1, 1)
        n_samples = len(magnitudes)
        if n_samples >= 4:  
            gmm = GaussianMixture(n_components=4)
        else:  # Utilise un nombre de composants égal au nombre d'échantillons si peu d'échantillons
            gmm = GaussianMixture(n_components=n_samples) if n_samples > 1 else None
        
        if gmm:  # Vérifie si un modèle GMM a été instancié
            gmm.fit(magnitudes)
            # Récupère la moyenne de chaque composant du GMM
            component_means = gmm.means_.flatten()
            block_magnitudes[block] = np.mean(component_means)  # Calcule la moyenne des moyennes des composantes
        else:
            block_magnitudes[block] = None
    return block_magnitudes'''
def compute_magnitude_model(block_vectors, default_components=4):
    """
    Computes the average magnitude for each block using a GMM model trained on all block magnitudes.
    
    Args:
        block_vectors (dict): A dictionary where keys are block labels and values are lists of vectors.
                              The fourth element of each vector is treated as a magnitude.
        default_components (int): Default number of GMM components to use when training the model.
    
    Returns:
        dict: A dictionary with block labels as keys and the average magnitude for each block.
    """
    all_magnitudes = []
    
    # Collect all magnitudes from all blocks
    for vectors in block_vectors.values():
        all_magnitudes.extend([v[3] for v in vectors])
    
    all_magnitudes = np.array(all_magnitudes).reshape(-1, 1)
    n_samples = len(all_magnitudes)
    
    # Train a single GMM model on all magnitudes
    if n_samples >= default_components:
        gmm = GaussianMixture(n_components=default_components)
    elif n_samples > 1:
        gmm = GaussianMixture(n_components=n_samples)
    else:
        return {block: None for block in block_vectors}  # Return None for all blocks if not enough data
    
    gmm.fit(all_magnitudes)
    component_means = gmm.means_.flatten()
    
    # Compute and store the mean of component means for each block
    block_magnitudes = {}
    for block, vectors in block_vectors.items():
        block_magnitudes[block] = np.mean(component_means) if vectors else None
    
    return block_magnitudes, gmm



if __name__ == "__main__":
    # Exemple d'utilisation avec des vecteurs de mouvement simulés
    block_vectors = {
        (0, 0): [(0, 0, 0.1, 2.0), (0, 0, 1.5, 3.0)],  # Format : (x, y, angle, magnitude)
        (1, 0): [(0, 0, 2.2, 1.5), (0, 0, 3.1, 2.8)]
    }

    directional_probs = compute_directional_probabilities(block_vectors)

    # Affichage des densités de probabilité directionnelle par bloc
    for block, model in directional_probs.items():
        for block1, magnitude in block_magnitudes.items():
            if block1==block:
                print(f"Bloc {block}:")
                for theta in np.linspace(0, 2 * np.pi, 5):
                    print(f"P({theta:.2f}) = {model.predict_density(theta):.4f}")
                print(f"magnitude:   {magnitude}")