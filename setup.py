from setuptools import setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'deep_vital/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':
    setup(
        name='deep_vital',
        version=get_version(),
        description='Deep Vital Toolbox',
        long_description=readme(),
        long_description_content_type='text/markdonw',
        author='wang-tf',
        author_email='ternencewang2015@outlook.com',
        # packages=find_packages(),
        include_package_data=True,
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        ext_modules=[],
        zip_safe=False
    )
    
