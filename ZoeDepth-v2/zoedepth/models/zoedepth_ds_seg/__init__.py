

from .zoedepth_ds_seg_v1 import ZoeDepth_DS_Seg

all_versions = {
    "v1": ZoeDepth_DS_Seg,
}

get_version = lambda v : all_versions[v]