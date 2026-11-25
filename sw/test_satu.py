import numpy as np, pandas as pd
from golden import Params, infer_batch

P = Params.from_files("../rtl/mem")
df = pd.read_csv("../data/bangla.csv")
print("shape:", df.shape)  # patut (28000, 784)

# Semua kolom ni pixel, bukan label
X = df.values  # (28000, 784)
bits = (X.reshape(-1, 28, 28) > 127).astype(np.uint8)

# Teka je, takde label nak banding
for s in range(0, 5):
    pred, _, _ = infer_batch(P, bits[s:s+1])
    print(f"Gambar {s}: model teka {pred[0]}")