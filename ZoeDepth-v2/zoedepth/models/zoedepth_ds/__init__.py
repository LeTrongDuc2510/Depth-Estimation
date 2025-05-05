

from .zoedepth_ds_v1 import ZoeDepth_DS

all_versions = {
    "v1": ZoeDepth_DS,
}

get_version = lambda v : all_versions[v]