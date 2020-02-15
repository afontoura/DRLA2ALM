from setuptools import setup

setup(name = 'DRLA2ALM',
      version = '0.1',
      description = 'Custom gym environment to model an Asset-Liability Management problem',
      url = 'https://github.com/afontoura/DRLA2ALM',
      author = 'Alan Fontoura',
      author_email = 'alan.fontoura@eic.cefet-rj.br',
      install_requires = ['gym', 'numpy', 'pandas', 'scipy']
)
