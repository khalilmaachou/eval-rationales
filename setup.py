import pkg_resources
from setuptools import setup, find_packages

setup(
    name='evalrationales',
    description='An End-to-End Toolkit to Explain and Evaluate Transformers-Based Models',
    url='https://github.com/jeslev/eval-rationales',
    author='Khalil Maachou',
    author_email='khalilmaachou.99@gmail.com',
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],

    classifiers=[
        "Development Status :: 1 - Planning",
        "'Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"
    ],
)