from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='state_reconstruction',
      version='0.1dev',
      description='State reconstruction library',
      long_description=readme(),
      classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3',
      ],
      keywords='data analysis',
      url='https://github.com/david-wei/state_reconstruction',
      author='David Wei',
      author_email='david94.wei@gmail.com',
      license='GPLv3',
      packages=find_packages(),
      entry_points={
            'console_scripts': [
                  # 'script_name = package.module:function'
            ],
            'gui_scripts': []
      },
      install_requires=[
      ],
      python_requires='>=3.6',
      dependency_links=[
      ],
      include_package_data=True,
      zip_safe=False)
