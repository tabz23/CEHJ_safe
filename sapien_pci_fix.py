"""Unused V100 SAPIEN/Vulkan workarounds.

These used to rewrite CUDA/NVIDIA visibility and patch libsapien in memory.
That path is disconnected: renderer init is stock `sapien.SapienRenderer()`,
and this module is a no-op so it cannot change GPU isolation on H100.
"""


def prepare_sapien_gpu_env() -> None:
    return


def make_sapien_renderer():
    import sapien

    return sapien.SapienRenderer()


def install_sapien_pci_hook() -> None:
    return


def ensure_sapien_pci_preload() -> None:
    return
