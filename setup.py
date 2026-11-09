from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="nccl_ops",
    ext_modules=[
        CUDAExtension(
            "nccl_ops",
            ["csrc/nccl_ops.cpp"],
            libraries=["nccl"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
