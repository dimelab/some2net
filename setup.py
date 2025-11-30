"""Setup script for social-network-analytics package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='social-network-analytics',
    version='0.1.0',
    description='Social media network analytics with Named Entity Recognition',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Jakob Bk',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/social-network-analytics',
    license='MIT',

    # Package discovery
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    # Python version requirement
    python_requires='>=3.9',

    # Dependencies
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'pandas>=2.0.0',
        'networkx>=3.0',
        'streamlit>=1.25.0',
        'langdetect>=1.0.9',
        'sentencepiece>=0.1.99',
        'numpy>=1.24.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
        'plotly>=5.14.0',
        # fa2 removed - using front-end Sigma.js for Force Atlas 2 visualization
        'diskcache>=5.6.0',
        'chardet>=5.0.0',
    ],

    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'pylint>=2.17.0',
            'mypy>=1.0.0',
        ],
        'advanced': [
            'sentence-transformers>=2.2.0',
            'requests>=2.28.0',
        ],
    },

    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'sna-web=cli.app:main',
            'sna-cli=cli.cli:main',
        ],
    },

    # Package classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],

    # Keywords for discovery
    keywords='social-media network-analysis ner named-entity-recognition nlp network-science',

    # Include package data
    include_package_data=True,

    # Package data
    package_data={
        '': ['*.yaml', '*.yml', '*.json'],
    },

    # Zip safe
    zip_safe=False,
)
