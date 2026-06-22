import cv2
import numpy as np

ground_truth = cv2.imread(
    "data/processed/train/rgb/rgb_0.png"
)

prediction = cv2.imread(
    "outputs/prediction.png"
)

mse = np.mean(
    (ground_truth.astype("float") -
     prediction.astype("float")) ** 2
)

if mse == 0:
    print("PSNR = Infinite")
else:
    psnr = 20 * np.log10(
        255.0 / np.sqrt(mse)
    )

    print("PSNR:", round(psnr, 2), "dB")