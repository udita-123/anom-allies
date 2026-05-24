from pathlib import Path

fpath = Path(r"C:/Users/KIIT0001/AppData/Local/Programs/Python/Python312/Lib/site-packages/deep_sort_realtime/embedder/embedder_pytorch.py")
src = fpath.read_text(encoding="utf-8")

# Print first 20 lines so we can see exact content
lines = src.splitlines()
for i, line in enumerate(lines[:20], 1):
    print(f"{i:3}: {line}")

# Patch: replace the broken import and resource_filename calls
src_new = src.replace(
    "import importlib.metadata as pkg_resources",
    "from pathlib import Path as _Path"
)
src_new = src_new.replace(
    'MOBILENETV2_BOTTLENECK_WTS = pkg_resources.resource_filename(\n    "deep_sort_realtime", "embedder/weights/mobilenetv2_bottleneck_wts.pt"\n)',
    "MOBILENETV2_BOTTLENECK_WTS = str(_Path(__file__).parent / 'weights' / 'mobilenetv2_bottleneck_wts.pt')"
)
src_new = src_new.replace(
    'TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS = pkg_resources.resource_filename(\n    "deep_sort_realtime", "embedder/weights/osnet_ain_ms_d_c_wtsonly.pth"\n)',
    "TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS = str(_Path(__file__).parent / 'weights' / 'osnet_ain_ms_d_c_wtsonly.pth')"
)

if src_new != src:
    fpath.write_text(src_new, encoding="utf-8")
    print("\nPatched successfully.")
else:
    print("\nNothing changed — printing full file for inspection:")
    print(src)