from setuptools import setup

setup(
    name='mbeann',
    version='0.1.0',
    author='Motoaki Hiraga',
    author_email='100577843+motoHiraga@users.noreply.github.com',
    url='https://github.com/motoHiraga/pyMBEANN',
    license='MIT',
    description='An implementation of Mutation-Based Evolving Artificial Neural Network (MBEANN)',
    py_modules=['mbeann'],
    install_requires=['numpy', 'networkx', 'matplotlib',],
    extras_require={'examples': ['pandas']}
)
