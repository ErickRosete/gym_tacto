from setuptools import setup

setup(name='gym_tacto',
      version='0.0.1',
      description='Environments for robotic manipulation',
      author='Erick Rosete Beas',
      author_email='erickrosetebeas@hotmail.com',
      install_requires=['hydra.core(>=1.0.6)',
                        'gym',
                        'pybulletX',
                        'tacto']
)