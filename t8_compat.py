"""对第三方 T8 包的运行时兼容补丁（不改其源文件，只在内存里打补丁）。

背景
----
comfyui-minimax-h3-audio-T8 的 **Hybrid**（首尾帧 + 参考图同时存在）兼容性探测
在 conditioning.py 中用非法索引构造了一个假 keyframe：

    keyframe = {"resolved_frame_index": 2, ...}          # 探测用
    build_packed_layout(..., keyframes=[keyframe], frame_count=5)

而当前 ComfyUI 的 PackedLayout.__init__（comfy/ldm/minimax/model.py）只接受
0（首帧）或 frame_count-1（尾帧）作为锚点索引，其余一律抛：

    ValueError: only first/last keyframe anchors are supported

探测因此必然失败，T8 会误判「PackedLayout 实现不兼容」并禁用 Hybrid 路径，表现为：

    RuntimeError: The active MiniMax H3 PackedLayout implementation rejected
    the guarded legacy Hybrid compatibility probe.

而 T8 的真实生成路径用的正是 0 与 frame_count-1（conditioning.py 里首帧写 0、
尾帧写 frame_count-1），**只有探测那处数据写错了**。

做法
----
把 T8 conditioning 模块的 build_packed_layout 包一层：遇到既不是 0、也不是
frame_count-1 的 keyframe 索引时，改写为 0 再交给原函数。

为什么安全
----------
1. build_packed_layout 在 conditioning.py 里只被探测函数
   （assert_hybrid_layout_contract / _bypass_verified_obsolete_layout_patch）调用，
   真实生成路径不使用它；
2. 真实路径的索引本就合法，因此包装对真实生成零影响；
3. 上游 T8 修好之后，非法索引不再出现，包装自动退化为透明传递。

本模块全部代码位于本仓库，T8 作者更新不会覆盖；每次节点执行时幂等地重新确认。
"""

import importlib
import logging
import sys

_PATCH_FLAG = "_hjl_hybrid_probe_fix_applied"


def _is_t8_conditioning_module(mod) -> bool:
    """严格判定：必须是 conditioning.py，且 assert_hybrid_layout_contract
    的全局命名空间里确实有 build_packed_layout。

    只用 hasattr 判定会误命中 torch.ops / torch.classes 这类对任意属性名
    都返回 True 的动态命名空间对象（实测踩过），因此必须加这两道校验。
    """
    if mod is None or not getattr(mod, "__file__", None):
        return False
    if not str(mod.__file__).replace("\\", "/").endswith("conditioning.py"):
        return False
    probe = getattr(mod, "assert_hybrid_layout_contract", None)
    if not callable(probe):
        return False
    # 关键：探测函数内部是通过模块全局查找 build_packed_layout 的，
    # 只有它出现在 __globals__ 里，替换模块属性才会真正生效。
    return "build_packed_layout" in getattr(probe, "__globals__", {})


def _find_t8_conditioning_module():
    """定位 T8 的 conditioning 模块：先按节点类反查，再退回严格扫描 sys.modules。"""
    try:
        import nodes as comfy_nodes

        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3AudioConditioningT8")
        if cls is not None:
            pkg = cls.__module__.rsplit(".", 1)[0]
            mod = importlib.import_module(pkg + ".conditioning")
            if _is_t8_conditioning_module(mod):
                return mod
    except Exception:
        pass

    for mod in list(sys.modules.values()):
        if _is_t8_conditioning_module(mod):
            return mod
    return None


def ensure_t8_hybrid_probe_fix() -> bool:
    """为 T8 的 Hybrid 探测打上兼容补丁（幂等）。

    返回 True 表示补丁已就绪；False 表示未找到 T8 模块（未安装或尚未加载）。
    """
    mod = _find_t8_conditioning_module()
    if mod is None:
        return False
    if getattr(mod, _PATCH_FLAG, False):
        return True

    orig = getattr(mod, "build_packed_layout", None)
    if orig is None:
        return False

    def build_packed_layout(text_len, latent_t, latent_h, latent_w, audio_t,
                            keyframes=None, refs=None, frame_count=None):
        if keyframes:
            fixed, changed = [], False
            for kf in keyframes:
                try:
                    idx = kf.get("resolved_frame_index")
                except AttributeError:
                    fixed.append(kf)
                    continue
                # 非法锚点索引（既非首帧也非尾帧）按首帧处理 —— 仅探测会出现
                if idx is not None and idx != 0 and (
                        frame_count is None or idx != frame_count - 1):
                    kf = dict(kf)
                    kf["resolved_frame_index"] = 0
                    changed = True
                fixed.append(kf)
            if changed:
                keyframes = fixed
        return orig(text_len, latent_t, latent_h, latent_w, audio_t,
                    keyframes=keyframes, refs=refs, frame_count=frame_count)

    mod.build_packed_layout = build_packed_layout
    setattr(mod, _PATCH_FLAG, True)
    logging.info("[HJL] T8 Hybrid 探测兼容补丁已生效（非法 keyframe 索引按首帧处理）")
    return True
