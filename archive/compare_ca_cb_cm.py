import logging

from .analysis import plot_target_vs_reference
from .globals import DATASETS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    for target in DATASETS['pfam'][:20]:
        plot_target_vs_reference(target,3)
        plot_target_vs_reference(target,2)


if __name__ == "__main__":
    main()
