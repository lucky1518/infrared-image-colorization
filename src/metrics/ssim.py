import cv2

from skimage.metrics import structural_similarity as ssim

ground_truth = cv2.imread(
    "data/processed/train/rgb/rgb_0.png"
)

prediction = cv2.imread(
    "outputs/prediction.png"
)

score = ssim(
    ground_truth,
    prediction,
    channel_axis=2
)

print("SSIM:", round(score, 4))