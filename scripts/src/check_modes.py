import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
modes = Counter()
tinted = 0
for path in root.rglob("*.jp*g"):
  with Image.open(path) as image:
    modes[image.mode] += 1
    if image.mode == "RGB":
      pixels = np.asarray(image, dtype=np.int16)
      spread = max(
          np.abs(pixels[..., 0] - pixels[..., 1]).max(),
          np.abs(pixels[..., 1] - pixels[..., 2]).max(),
      )
      tinted += spread > 2
print(dict(modes))
print("RGB files with channels differing by more than 2 levels:", tinted)