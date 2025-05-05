from .dpt_ds_v1 import DPT_DeepSup 

all_versions = {
    "v1": DPT_DeepSup,
}

get_version = lambda v : all_versions[v]