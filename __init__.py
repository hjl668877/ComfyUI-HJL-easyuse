from .nodes import NODE_CLASS_MAPPINGS as _HJL_CORE
from .h3_two_pass import NODE_CLASS_MAPPINGS as _HJL_TWO_PASS

NODE_CLASS_MAPPINGS = {**_HJL_CORE, **_HJL_TWO_PASS}

# 在 UI 里显示的名字
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3_EditConditioningWH": "H3 Edit Conditioning W/H (HJL)",
    "EditConditioningWH": "Edit Conditioning W/H (HJL, legacy)",
    "H3_TwoPassSampler": "H3 Two-Pass Sampler (HJL)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
