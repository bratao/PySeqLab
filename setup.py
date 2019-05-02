'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''

from setuptools import setup


# package description
expl_files = ['README.rst']
long_description = '\n'.join([open(f, 'r').read() for f in expl_files])
curr_version = open('version.txt', 'r').read().rstrip()

if __name__ == "__main__":
    setup(name='PySeqLab',
          version=curr_version,
          description='A package for performing structured prediction (i.e.sequence labeling and segmentation).',
          long_description=long_description,
          author="Ahmed Allam",
          author_email='ahmed.allam@yale.edu',
          license="MIT",
          url='https://bitbucket.org/A_2/pyseqlab/',
          download_url='https://bitbucket.org/A_2/pyseqlab/downloads',
          keywords='conditional random fields, semi-markov conditional random fields, structured prediction, sequence labeling and segmentation, bioinformatics',
          packages=["pyseqlab"],
          install_requires=["numpy>=1.8.0", "scipy>=0.13"],
          classifiers=['Development Status :: 4 - Beta',
                       'Environment :: Console',
                       'Intended Audience :: Science/Research',
                       'License :: OSI Approved :: MIT License',
                       'Natural Language :: English',
                       'Operating System :: OS Independent',
                       'Programming Language :: Python :: 3',
                       'Programming Language :: Python :: 3.4',
                       'Programming Language :: Python :: 3.5',
                       'Topic :: Scientific/Engineering :: Bio-Informatics',
                       'Topic :: Scientific/Engineering :: Information Analysis'])
    
