import cv2
import numpy as np
import matplotlib.pyplot as plt
from detection.point_detection import detect_harris_points, track_points, divide_into_blocks
from grouping.block_grouping import group_blocks, visualize_similar_blocks
from detection.magnitude_direction_models import compute_directional_probabilities, compute_magnitude_model
from grouping.group_tracking import Tracker
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
        frame_with_groups = visualize_similar_blocks(curr_frame, block_groups)

# Display the frame
        plt.imshow(cv2.cvtColor(frame_with_groups, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axes for a clean display
        plt.show()
        # Get the previous frame's block groups for tracking
        prev_block_groups = video_frames.get(idx - 1, {}).get('block_groups', [])
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
        video_frames[idx] = {'frame': curr_frame, 'block_groups': block_groups}
        prev_frame = curr_frame
        idx += 1 #######
        