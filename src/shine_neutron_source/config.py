"""The user's configuration file: where their nuclear-data libraries are.

Library locations belong to the machine, not to a case, so they live in a per-user TOML file,
~/.shine-neutron-source.toml (or the file named by the SHINE_NEUTRON_SOURCE_CONFIG
environment variable), with a [libraries] table naming directories of evaluated files and a
[defaults] nuclear_data list. Its format and use are described in data/README.md
("Evaluated nuclear data libraries").
"""

import os
import tomllib

CONFIG_ENV = "SHINE_NEUTRON_SOURCE_CONFIG"
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".shine-neutron-source.toml")

# The nuclear_data entry for the built-in DROSG-2000 two-body reactions (nuclear.REACTIONS)
DROSG = "drosg"
# nuclear_data when neither the case nor the configuration gives one
DEFAULT_NUCLEAR_DATA = [DROSG]


class ConfigError(ValueError):
    """The configuration file is invalid, or names a library that is not defined."""


def config_path():
    """Return the configuration file's path: SHINE_NEUTRON_SOURCE_CONFIG if set, else
    ~/.shine-neutron-source.toml. The file need not exist."""
    return os.environ.get(CONFIG_ENV) or CONFIG_FILE


def load_config(path=None):
    """Return {"path", "libraries": {name: directory}, "nuclear_data": default list or None}
    from the configuration file (empty if there is none). Raises ConfigError if it is
    invalid."""
    path = config_path() if path is None else str(path)
    config = {"path": path, "libraries": {}, "nuclear_data": None}
    if not os.path.isfile(path):
        return config
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    unknown = sorted(set(data) - {"libraries", "defaults"})
    if unknown:
        raise ConfigError(f"{path}: unknown table(s) {', '.join(unknown)}")
    base = os.path.dirname(os.path.abspath(path))
    for name, directory in data.get("libraries", {}).items():
        if not isinstance(directory, str):
            raise ConfigError(f"{path}: [libraries] {name} must be a directory, got {directory!r}")
        if name == DROSG:
            raise ConfigError(f"{path}: [libraries] cannot define {DROSG!r}, the built-in data")
        directory = os.path.expanduser(directory)
        config["libraries"][name] = os.path.normpath(os.path.join(base, directory))
    defaults = data.get("defaults", {})
    unknown = sorted(set(defaults) - {"nuclear_data"})
    if unknown:
        raise ConfigError(f"{path}: unknown setting(s) in [defaults]: {', '.join(unknown)}")
    if "nuclear_data" in defaults:
        entries = defaults["nuclear_data"]
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ConfigError(f"{path}: [defaults] nuclear_data must be a list of names")
        config["nuclear_data"] = list(entries)
    return config


def resolve_nuclear_data(entries=None, base_dir=None, config=None):
    """Return (entries, directories): the nuclear_data entries as given (the configuration's
    default, or ["drosg"], if entries is None) and what each resolves to: "drosg", the
    directory of a library named in the configuration, or a directory path (relative to
    base_dir, if given, else the working directory). A name defined in the configuration
    wins over a directory of the same name.

    Raises ConfigError for an entry that is neither a defined name nor an existing directory.
    """
    config = load_config() if config is None else config
    if entries is None:
        entries = config["nuclear_data"] or DEFAULT_NUCLEAR_DATA
    if isinstance(entries, (str, os.PathLike)):
        entries = [entries]
    entries = [str(e) for e in entries]
    directories = []
    for entry in entries:
        if entry == DROSG:
            directories.append(DROSG)
        elif entry in config["libraries"]:
            directories.append(config["libraries"][entry])
        else:
            directory = os.path.expanduser(entry)
            if base_dir is not None and not os.path.isabs(directory):
                directory = os.path.join(base_dir, directory)
            directories.append(os.path.normpath(directory))
        if directories[-1] != DROSG and not os.path.isdir(directories[-1]):
            names = ", ".join(sorted(config["libraries"])) or "none"
            raise ConfigError(
                f"nuclear_data entry {entry!r} is not {DROSG!r}, a library defined in "
                f"{config['path']} (defined: {names}) or an existing directory "
                f"({directories[-1]})"
            )
    return entries, directories
