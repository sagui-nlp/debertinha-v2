import os


def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    nanochat_dir = os.path.join(cache_dir, "cc-dataset")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir
