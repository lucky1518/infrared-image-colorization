from dataset import LandsatDataset

dataset = LandsatDataset(
    "data/processed/train/input",
    "data/processed/train/rgb"
)

print("Dataset Size:", len(dataset))

x, y = dataset[0]

print("Input Shape:", x.shape)
print("Target Shape:", y.shape)