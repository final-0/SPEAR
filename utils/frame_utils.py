import re
from os.path import splitext

import h5py
import numpy as np
from PIL import Image


def readPFM(file):
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', f.readline())
        if not dim_match:
            raise Exception('Malformed PFM header.')
        width, height = map(int, dim_match.groups())
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
    data = np.reshape(data, (height, width, 3) if color else (height, width))
    return np.flipud(data)


def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]


def writeFlo5File(flow, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)


def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext in ('.png', '.jpeg', '.ppm', '.jpg'):
        return Image.open(file_name)
    if ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        return flow if flow.ndim == 2 else flow[:, :, :-1]
    if ext == '.flo5':
        return readFlo5Flow(file_name)
    raise ValueError(f"unsupported file type: {file_name}")
