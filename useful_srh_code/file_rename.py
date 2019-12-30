
# example file name: "/home/todd/Desktop/411-5238/411-5238_3"

import os

for root, dirs, files in os.walk(""):
    if not files:
        continue
    prefix = os.path.basename(root)
    for f in files:
        os.rename(os.path.join(root, f), os.path.join(root, "{}_{}".format(prefix, f)))
