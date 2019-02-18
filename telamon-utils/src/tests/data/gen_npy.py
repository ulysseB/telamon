import os
import re

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_CASE_RE = re.compile(
    r'^\s*\[(?P<names>[^]]+)]\s*"(?P<python>[^"]+)"')

def extract_test_cases(lines):
    for line in lines:
        if line.strip() == '// @NPY_START':
            break
    else:
        # No test cases found on this file
        return

    for line in lines:
        match = TEST_CASE_RE.match(line)
        if not match:
            if line.strip() == '// @NPY_END':
                # There may be extra blocks later
                yield from extract_test_cases(lines)
                return
            else:
                continue

        names, python = match.group('names', 'python')

        for name_ty in names.split(','):
            if ':' not in name_ty:
                continue

            name, rust_ty = name_ty.split(':')
            if rust_ty.startswith('i'):
                dtype = 'np.int{}'.format(int(rust_ty[1:]))
            elif rust_ty.startswith('u'):
                dtype = 'np.uint{}'.format(int(rust_ty[1:]))
            elif rust_ty.startswith('f'):
                dtype = 'np.float{}'.format(int(rust_ty[1:]))
            else:
                raise RuntimeError("Unknown NumPy type mapping for Rust type: `{}`".format(rust_ty))

            yield (name.strip(), dtype, python)

def main(fnames):
    for fname in fnames:
        with open(fname, 'r') as f:
            for name, dtype, python in extract_test_cases(f):
                print("Generating {}.npy...".format(name))

                with open(os.path.join(BASE_DIR, name + '.npy'), 'wb') as out:
                    np.save(out, eval(python.replace('?', dtype), {'np': np}))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fname', nargs='+', help='name of the rust file to parse for test cases')

    args = parser.parse_args()

    main(fnames=args.fname)
