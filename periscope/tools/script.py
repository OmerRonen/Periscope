import os
import shutil

from periscope.data.creator import DataCreator
from periscope.data.property import _get_property_path
from periscope.utils.constants import PATHS
from periscope.utils.drive import upload_folder
from periscope.utils.utils import check_path


def main():
    family = "xcl1_family"
    dc = DataCreator("A0A2K6CRQ6", family=family)
    msa = dc._parse_msa()
    property_path = os.path.join(PATHS.periscope, "property")
    check_path(property_path)
    for t in msa:

        try:
            _ = DataCreator(t, family=family, train=False).raptor_properties

            pth = _get_property_path(t, family)
            dst = os.path.join(property_path, t)
            shutil.copytree(pth, dst)

            upload_folder(dst, dst.split('Periscope/')[-1])
        except Exception:
            pass


if __name__ == '__main__':
    main()
