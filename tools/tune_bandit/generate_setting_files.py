#!/usr/bin/env python3
import os

telamon_root = os.path.realpath("../../")
tuning_path = os.path.realpath(".")
setting_path = tuning_path + "/settings/"

spec = {
    "log_file": str,
    "num_workers": int,
    "stop_bound": float,
    "timeout": float,
    "distance_to_best": float,
    "algorithm": {
        "type": ("bandit", ),
        "new_nodes_order": ("api", "random", "bound", "weighted_random"),
        "old_nodes_order": ("bound", "bandit", "weighted_random"),
        "threshold": int,
        "delta": float,
        "monte_carlo": bool,
    },
}

def check(value, *, path=(), spec=spec):
    """Check that a value adheres to a specification.

    Entries in the specification can be:
      - A tuple of allowed values: the provided `value` must be one of these
      - A type: the provided `value` must have that type
      - A dict: the provided `value` must be a dict which has only keys defined
        in the `spec` (some of the `spec` keys may be missing from the `value`
        dict). Each provided value in the dict will be checked recursively with
        the corresponding entry in the spec.

    All entries in `value` are optional (i.e. can be `None`), unless the
    corresponding entry in the specification is a `dict`.
    """

    if isinstance(spec, dict):
        if not isinstance(value, dict):
            print("Key {} should be a dict; got {}".format(
                ".".join(path), value))
            return False
        unknown = set(value.keys()) - set(spec.keys())
        if unknown:
            print("Keys {} are missing".format(
                ", ".join(".".join(path + (key, )) for key in unknown)))
        return all(
            check(value.get(key), path=path + (key, ), spec=spec_value)
            for key, spec_value in spec.items())
    elif value is None:
        return True
    elif isinstance(spec, type):
        if not isinstance(value, spec):
            print("Key {} should be a {}; got {!r}".format(
                ".".join(path), spec.__name__, value))
            return False
        else:
            return True
    elif isinstance(spec, tuple):
        if value not in spec:
            print("Key {} should be one of {}; got {!r}".format(
                ".".join(path), ", ".join(map(repr, spec)), value))
            return False
        return True
    else:
        raise AssertionError(
            "Invalid spec: {}".format(spec))

def serialize_value(value):
    """Serialize a single value.

    This is used instead of a single-shot `.format()` call because some values
    need special treatment for being serialized in YAML; notably, booleans must
    be written as lowercase strings, and floats exponents must not start with a
    0.
    """

    if isinstance(value, bool):
        return repr(value).lower()
    elif isinstance(value, float):
        return "{0:.16}".format(value).replace("e+0", "e+").replace("e-0", "e-")
    else:
        return repr(value)

def serialize(f, key, value):
    """Serialize a (key, value) pair into a file."""

    if isinstance(value, dict):
        f.write("[{}]\n".format(key))
        for k, v in value.items():
            serialize(f, k, v)
    elif value is not None:
        f.write("{} = {}\n".format(key, serialize_value(value)))

def create_setting_file(options_dict, filename):
    if not check(options_dict):
        print("Invalid Options dict")
        return

    with open(filename, "w+") as f:
        for key, value in options_dict.items():
            serialize(f, key, value)

def clear_directory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

filename = "test_py.log"
opts = {
    "num_workers": 24,
    "timeout": 150.,
    "algorithm": {
        "type": "bandit",
        "monte_carlo": True,
    }
}

if __name__ == "__main__":
    if not os.path.exists(setting_path):
        os.makedirs(setting_path)
    clear_directory(setting_path)
    for i in range(8):
        opts["algorithm"]["delta"] = pow(2, i) * 0.00001
        for j in range(1, 4):
            opts["algorithm"]["threshold"] = j * 10
            filename = ("d" + "-" + "{:3e}".format(opts["algorithm"]["delta"]) + "_" + "t"
                    + "{}".format(opts["algorithm"]["threshold"]) +".toml")
            create_setting_file(opts, setting_path + filename)
