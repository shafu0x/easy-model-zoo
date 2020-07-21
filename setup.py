from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='easy-model-zoo',
      version='0.2.4',
      description='Easily run deep learning models.',
      url='sharifelfouly.com',
      author='Sharif Elfouly',
      author_email='selfouly@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['torch==1.4.0',
                        'numpy',
                        'Pillow',
                        'torchvision==0.5',
                        'opencv-python',
                        'gdown',
                        'webcolors',
                        'pycocotools',
                        'bounding-box==0.1.3'
                        ])