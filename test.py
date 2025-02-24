  # OpenCV for handling video frames


# Example usage:
# Assuming you have a video frame `frame` and the block probabilities and magnitudes calculated
# You would first compute the block groups
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict
from detection.magnitude_direction_models import VonMisesMixture
import cv2  # OpenCV for handling video frames

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

# Example usage:
# Assuming you have a video frame `frame` and the block probabilities and magnitudes calculated
# You would first compute the block group
###################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict





# Example usage:
# Assuming `frame1`, `frame2`, and `block_groups1`, `block_groups2` are the two frames and their detected block groups.





##################################################
# Display the frame
import cv2
import numpy as np
from detection.point_detection import detect_harris_points, track_points, divide_into_blocks
from grouping.block_grouping import group_blocks
from detection.magnitude_direction_models import compute_directional_probabilities, compute_magnitude_model
from events.event_detection import detect_crowd_events
if __name__ == "__main__":
    print("Lancement de l'analyse vidéo...")

    # Charger la vidéo
    video_path = "/home/assioui/Downloads/video2.mp4" 
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la vidéo.")
        exit()

    points = detect_harris_points(prev_frame)
    video_frames = {}
    idx = 0

    # Initialize the tracker
    tracker = Tracker(min_distance=5)

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret or points is None:
            break

        # Track points between frames
        vectors, points, mvt_vector = track_points(prev_frame, curr_frame, points)
        
        # Divide into blocks
        blocks = divide_into_blocks(curr_frame, vectors)

        # Visualisation des points suivis
        for x, y, _, _ in vectors:
            cv2.circle(curr_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow("Tracking", curr_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Compute directional probabilities and magnitudes
        directional_probs = compute_directional_probabilities(blocks)
        block_magnitudes, gmm = compute_magnitude_model(blocks)

        # Group blocks based on directional probabilities and magnitudes
        block_groups = group_blocks(directional_probs, block_magnitudes)
        
        # Get the previous frame's block groups for tracking
        """prev_block_groups = video_frames.get(idx - 1, {}).get('block_groups', [])
        prev_frame = video_frames.get(idx - 1, {}).get('frame', prev_frame)

        if prev_block_groups:
            matches = tracker.update_groups(block_groups)
            frame1_with_blocks, frame2_with_blocks = tracker.visualize_tracked_blocks(
                prev_frame, curr_frame, matches, prev_block_groups, block_groups)

            # Display the results
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(frame1_with_blocks, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Frame 1 with Tracked Blocks')
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(frame2_with_blocks, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Frame 2 with Tracked Blocks')
            axes[1].axis('off')

            plt.show()

        # Store the current frame and block groups
        video_frames[idx] = {'frame': curr_frame, 'block_groups': block_groups}"""
        prev_frame = curr_frame
        idx += 1 #######
        