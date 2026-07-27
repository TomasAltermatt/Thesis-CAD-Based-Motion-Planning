from setuptools import setup, find_packages

setup(
    name='bbox',
    version='1.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'bpy',
        'mathutils',
        'trimesh',
        'jaxtyping',
    ],
)
