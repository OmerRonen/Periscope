import os
import matplotlib.pyplot as plt

from globals import periscope_path
from utils import parse_rr_file


cm = parse_rr_file(os.path.join(periscope_path, 'data', 'contactmap.txt'))
plt.matshow(cm)
