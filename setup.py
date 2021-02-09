from setuptools import setup

setup(name='image_perturbation',
      version='0.2.1',
      description='Image Buffer with Faster R-CNN module.',
      url='http://github.com/carlosejimenez/imageperturbation',
      author='Carlos E. Jiménez',
      author_email='cjsaltlake@gmail.com',
      license='MIT',
      packages=['image_perturbation'],
      install_requires=[
          'requests',
          'tqdm',
          'numpy',
          'filelock',
          'Pillow',
          'PyYAML',
          'scikit-image',
          'torch',
          'torchvision',
          'wget'
          ],
      zip_safe=False)
