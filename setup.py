from setuptools import setup, find_packages


VERSION = '0.0.1dev'

with open('README.md', 'r') as fp:
    LONG_DESCRIPTION = fp.read()

setup_info = dict(
    name='seq2seq-chatbot',
    version=VERSION,
    author='Tamara Katic',
    author_email='tamarakatic@gmail.com',
    url='https://github.com/tamarakatic/seq2seq',
    description='Chat bot based on seq2seq',
    long_description=LONG_DESCRIPTION,
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow==1.5.0',
        'tensorlayer==1.6.3',
        'nltk',
        'numpy'
    ]
)

setup(**setup_info)
