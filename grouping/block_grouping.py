import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict
from detection.magnitude_direction_models import VonMisesMixture
import cv2

def group_blocks(block_probabilities, block_magnitudes, threshold_angle=0.3, threshold_magnitude=1.0):
    """
    Regroupe les blocs ayant des directions et des magnitudes similaires.

    Args:
        block_probabilities (dict): Dictionnaire des blocs avec les modèles de von Mises.
        block_magnitudes (dict): Dictionnaire des blocs avec leur magnitude moyenne.
        threshold_angle (float): Différence maximale d'angle (en radians) pour appartenir au même groupe.
        threshold_magnitude (float): Différence maximale de magnitude pour appartenir au même groupe.

    Returns:
        List[set]: Liste des groupes de blocs. Chaque groupe est un ensemble de positions de blocs.
    """
    def are_blocks_similar(block1, block2):
        """Vérifie si deux blocs ont des magnitudes et directions similaires."""
        direction1 = block_probabilities[block1].means[0]  # Utilise la direction principale (μ0)
        direction2 = block_probabilities[block2].means[0]
        magnitude1 = block_magnitudes[block1]
        magnitude2 = block_magnitudes[block2]
        angle_diff = min(abs(direction1 - direction2), 2 * np.pi - abs(direction1 - direction2))
        if magnitude1 is None or magnitude2 is None:
            return False  # Ou une autre action selon le besoin
        return angle_diff < threshold_angle and abs(magnitude1 - magnitude2) < threshold_magnitude

    visited = set()
    groups = []

    for block in block_probabilities:
        if block in visited:
            continue

        group = set([block])
        stack = [block]
        visited.add(block)

        while stack:
            current_block = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Voisinage 4-connecté
                neighbor = (current_block[0] + dx, current_block[1] + dy)
                if neighbor in block_probabilities and neighbor not in visited:
                    if are_blocks_similar(current_block, neighbor):
                        stack.append(neighbor)
                        visited.add(neighbor)
                        group.add(neighbor)

        groups.append(group)

    return groups

def visualize_similar_blocks(frame, block_groups, block_size=(16, 16), color_map=None):
    """
    Visualizes the similar blocks by highlighting them on the frame.

    Args:
        frame (np.array): The video frame (image).
        block_groups (list): List of sets containing block positions grouped together.
        block_size (tuple): Size of each block (height, width).
        color_map (dict): Optional dictionary mapping group indices to colors.

    Returns:
        np.array: Frame with visualized blocks.
    """
    frame_copy = frame.copy()

    # Define color map if none is provided
    if color_map is None:
        color_map = {i: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
                     for i in range(len(block_groups))}
    
    for group_idx, group in enumerate(block_groups):
        color = color_map[group_idx]
        for block in group:
            top_left = (block[1] * block_size[1], block[0] * block_size[0])
            bottom_right = ((block[1] + 1) * block_size[1], (block[0] + 1) * block_size[0])
            cv2.rectangle(frame_copy, top_left, bottom_right, color, 2)

    return frame_copy

if __name__ == "__main__":
    # Exemple d'utilisation
    example_block_probabilities = {
        (0, 0): VonMisesMixture(n_components=4),  # Simulé avec des directions définies ailleurs
        (0, 1): VonMisesMixture(n_components=4),
        (1, 0): VonMisesMixture(n_components=4),
        (1, 1): VonMisesMixture(n_components=4)
    }

    for model in example_block_probabilities.values():
        model.means = [0.1, 1.5, 3.0, 4.5]  # Exemple de directions principales simulées

    example_block_magnitudes = {
        (0, 0): 1.2,
        (0, 1): 1.0,
        (1, 0): 1.1,
        (1, 1): 3.5
    }

    grouped_blocks = group_blocks(example_block_probabilities, example_block_magnitudes)
    print("Groupes de blocs similaires:")
    for i, group in enumerate(grouped_blocks):
        print(f"Groupe {i + 1}: {group}")