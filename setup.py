from setuptools import setup

setup(
    name='sc2_academy',
    version='1.0',
    packages=['sc2_academy'],
    install_requires=['pygame==2.0.0.dev8', 'pysc2', 'tensorflow==2.3.0',
                      'tensorflow-probability==0.11.0',
                      'dm-reverb', 'tf-agents==0.6.0',
                      'gym', 'numpy', 'absl-py'],
    url='',
    license='',
    author='Christoph Priesner',
    author_email='',
    description=''
)
