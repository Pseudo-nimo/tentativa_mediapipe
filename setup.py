from setuptools import setup, find_packages

setup(
    name='is_skeletons_detector',
    version='0.1.0',
    description='MediaPipe Pose skeleton detection service for the IS ecosystem',
    url='https://github.com/Pseudo-nimo/tentativa_mediapipe',
    author='Pseudo-nimo',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'is-skeletons-detector-stream=is_skeletons_detector.stream:main',
            'is-skeletons-detector-rpc=is_skeletons_detector.rpc:main',
        ],
    },
    zip_safe=False,
    install_requires=[
        'is-wire>=1.1.2',
        'is-msgs>=1.1.8',
        'opencensus-ext-zipkin',
        'mediapipe>=0.10.0',
        'opencv-python>=4.5.0',
        'python-dateutil',
    ],
)
