try:
    from setuptools import setup #enables develop
except ImportError:
    from distutils.core import setup

setup(name='python_speech_features',
      version='0.6.1',
      description='Python Speech Feature extraction',
      author='James Lyons',
      author_email='james.lyons0@gmail.com',
      license='MIT',
      url='https://github.com/man1207/python_speech_features',
      packages=['python_speech_features'],
      install_requires=[
        'numpy',
        'scipy',
      ]
    )
