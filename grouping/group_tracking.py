import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict
class Tracker:
    def __init__(self, min_distance=5):
        """Initialise le tracker avec une distance minimale pour correspondance."""
        self.min_distance = min_distance
        self.previous_groups = []
        self.current_groups = []

    def compute_centroid(self, group):
        """Calcule le barycentre d'un groupe de blocs."""
        if not group:
            return None
        x_coords, y_coords = zip(*group)
        return (sum(x_coords) / len(group), sum(y_coords) / len(group))

    def update_groups(self, current_groups):
        """Met à jour la correspondance des groupes entre deux images."""
        self.previous_groups = self.current_groups
        self.current_groups = current_groups

        if not self.previous_groups:
            return [(i, None) for i in range(len(current_groups))]

        current_centroids = [self.compute_centroid(group) for group in self.current_groups]
        previous_centroids = [self.compute_centroid(group) for group in self.previous_groups]

        matches = []
        for i, prev_centroid in enumerate(previous_centroids):
            if prev_centroid is None:
                continue

            distances = [distance.euclidean(prev_centroid, curr_centroid) for curr_centroid in current_centroids]
            if distances:
                min_distance = min(distances)
                min_index = distances.index(min_distance)

                if min_distance < self.min_distance:
                    matches.append((i, min_index))
                else:
                    matches.append((i, None))

        new_groups = [(None, j) for j in range(len(current_centroids)) if all(j != m[1] for m in matches)]
        matches.extend(new_groups)

        return matches

    def visualize_tracked_blocks(self, frame1, frame2, matches, block_groups1, block_groups2, block_size=(16, 16)):
        frame1_copy = frame1.copy()
        frame2_copy = frame2.copy()

    # Generate a random color map for the matched blocks
        color_map = {}
        color_counter = 0
    
        for prev_idx, curr_idx in matches:
            if prev_idx is None or curr_idx is None:
                continue  # No valid match, so skip it

        # Ensure the indices are within bounds
            if prev_idx < len(block_groups1) and curr_idx < len(block_groups2):
                if (prev_idx, curr_idx) not in color_map:
                    color_map[(prev_idx, curr_idx)] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                color = color_map[(prev_idx, curr_idx)]

            # Draw rectangles on both frames for matched blocks
                for block in block_groups1[prev_idx]:
                    top_left = (block[1] * block_size[1], block[0] * block_size[0])
                    bottom_right = ((block[1] + 1) * block_size[1], (block[0] + 1) * block_size[0])
                    cv2.rectangle(frame1_copy, top_left, bottom_right, color, 2)

                for block in block_groups2[curr_idx]:
                    top_left = (block[1] * block_size[1], block[0] * block_size[0])
                    bottom_right = ((block[1] + 1) * block_size[1], (block[0] + 1) * block_size[0])
                    cv2.rectangle(frame2_copy, top_left, bottom_right, color, 2)

        return frame1_copy, frame2_copy


if __name__ == "__main__":
    tracker = Tracker(min_distance=5)

    # Exemple fictif de groupes de blocs sur deux images
    example_groups_frame_f = [
        {(0, 0), (0, 1), (1, 0)},  # Groupe 1
        {(5, 5), (5, 6), (6, 5)}   # Groupe 2
    ]

    example_groups_frame_f1 = [
        {(0, 1), (1, 1), (1, 2)},  # Groupe 1 correspondant
        {(6, 5), (6, 6), (7, 5)},  # Groupe 2 correspondant
        {(10, 10), (10, 11)}       # Nouveau groupe
    ]

    tracker.update_groups(example_groups_frame_f)
    matches_f_to_f1 = tracker.update_groups(example_groups_frame_f1)

    print("Correspondances des groupes (image f -> image f+1):")
    for match in matches_f_to_f1:
        print(f"Groupe {match[0]} -> Groupe {match[1]}")