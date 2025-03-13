from setuptools import setup, find_packages

setup(
    name="weightlifting-performance-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'mediapipe>=0.8.10',
        'matplotlib>=3.4.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'Pillow>=8.0.0',
        'seaborn>=0.11.0',
        'PySide6>=6.6.0',
    ],
    entry_points={
        'console_scripts': [
            'weightlifting-analyzer=src.ui.app:launch_app',
        ],
    },
    author="Andrea Bugeja",
    author_email="andrea.bugeja@hotmail.com",
    description="A Python application that processes weightlifting videos to automatically analyze and provide performance metrics",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    keywords="weightlifting, sports, performance-analysis, computer-vision",
    url="https://github.com/yourusername/weightlifting-performance-analyzer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Sports/Health",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
)