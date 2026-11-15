# -*- mode: python ; coding: utf-8 -*-
# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# PyInstaller recipe of the Windows build (a folder with MeshBench.exe):
#     pyinstaller --noconfirm packaging/meshbench.spec
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, APP)
from modules import branding  # noqa: E402  (name, version, author)

datas = [
    (os.path.join(APP, "ui", "main.ui"), "ui"),
    (os.path.join(APP, "resources"), "resources"),
    (os.path.join(ROOT, "models", "Stanford_Bunny.stl"), "models"),
    (os.path.join(ROOT, "LICENSE"), "."),
]
hiddenimports = collect_submodules("modules") + collect_submodules("vtkmodules")

a = Analysis(
    [os.path.join(APP, "main.py")],
    pathex=[APP],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "jupyter", "notebook",
              "pytest", "torch", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)

windows = sys.platform.startswith("win")
version_file = None
if windows:
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable,
        VarFileInfo, VarStruct, VSVersionInfo)
    numbers = tuple(int(n) for n in (branding.VERSION + ".0.0.0").split(".")[:4])
    version_file = VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("CompanyName", branding.AUTHOR),
                StringStruct("FileDescription", f"{branding.APP_NAME} - {branding.TAGLINE}"),
                StringStruct("FileVersion", branding.VERSION),
                StringStruct("InternalName", branding.APP_NAME),
                StringStruct("LegalCopyright", branding.COPYRIGHT),
                StringStruct("OriginalFilename", f"{branding.APP_NAME}.exe"),
                StringStruct("ProductName", branding.APP_NAME),
                StringStruct("ProductVersion", branding.VERSION),
            ])]),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ])

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=branding.APP_NAME,
    console=False,
    icon=os.path.join(APP, "resources", "meshbench.ico") if windows else None,
    version=version_file,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=branding.APP_NAME, upx=False)
