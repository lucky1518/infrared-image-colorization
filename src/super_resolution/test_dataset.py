from dataset import SRDataset

dataset = SRDataset(
    "data/processed/train_sr/input",
    "data/processed/train_sr/target"
)

print("=" * 50)
print("Dataset Size :", len(dataset))
print("=" * 50)

lr, hr = dataset[0]

print("Low Resolution :", lr.shape)
print("High Resolution:", hr.shape)