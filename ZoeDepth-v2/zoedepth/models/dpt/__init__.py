from .dpt_v1 import DPT

all_versions = {
    "v1": DPT,
}

get_version = lambda v : all_versions[v]