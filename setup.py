from setuptools import setup, find_packages

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]
setup(
    name='Online Content Popularity',
    version='0.0.1',
    author='Monal',
    author_email='monalmonu4@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ]
)