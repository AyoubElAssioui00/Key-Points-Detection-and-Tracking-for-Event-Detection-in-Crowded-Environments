import cv2
import numpy as np

def detect_harris_points(frame, block_size=2, ksize=3, k=0.04):
    """Détecte des points d'intérêt avec l'algorithme de Harris."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    harris_response = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    harris_response = cv2.dilate(harris_response, None)  # Dilate pour marquer les coins
    threshold = 0.01 * harris_response.max()
    keypoints = np.argwhere(harris_response > threshold)
    keypoints = [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints]
    return np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

def track_points(prev_frame, curr_frame, points, min_movement=0.5):
    """Suit les points d'intérêt avec le flux optique de Lucas-Kanade."""
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, points, None)
    valid_points = []
    movement_vectors = []

    for i, (new, old) in enumerate(zip(new_points, points)):
        if status[i] == 1:
            x1, y1 = old.ravel()
            x2, y2 = new.ravel()
            dx, dy = x2 - x1, y2 - y1
            movement_vectors += [(dx, dy)]
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude >= min_movement:
                angle = np.arctan2(dy, dx)
                valid_points.append((x2, y2, angle, magnitude))

    return np.array(valid_points, dtype=np.float32), new_points, movement_vectors


def divide_into_blocks(frame, points, block_size=(16, 16)):
    """Divise l'image en blocs et associe les vecteurs de mouvement aux blocs correspondants."""
    height, width = frame.shape[:2]
    blocks = {}  # Dictionnaire pour stocker les vecteurs par bloc

    for x, y, angle, magnitude in points:
        block_x = int(x // block_size[1])
        block_y = int(y // block_size[0])
        block_key = (block_y, block_x)

        if block_key not in blocks:
            blocks[block_key] = []

        blocks[block_key].append((x, y, angle, magnitude))

    return blocks

if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/assioui/Downloads/video2.mp4")
    ret, prev_frame = cap.read()
    points = detect_harris_points(prev_frame)

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret or points.size == 0:
            break

        vectors, points, m = track_points(prev_frame, curr_frame, points)
        blocks = divide_into_blocks(curr_frame, vectors)

        # Visualisation des points suivis
        for x, y, _, _ in vectors:
            cv2.circle(curr_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        cv2.imshow("Tracking", curr_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev_frame = curr_frame

    cap.release()
    cv2.destroyAllWindows()