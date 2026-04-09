from setuptools import setup, find_packages

setup(
    name='mediapipe',
    version='0.1.0',
    description='MediaPipe Pose skeleton detection service for the IS ecosystem',
    url='https://github.com/Pseudo-nimo/tentativa_mediapipe',
    author='Pseudo-nimo',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.0',
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
        'mediapipe',
        'opencensus-ext-zipkin==0.2.2',
        'opencv-python>=4.5.0',
        'python-dateutil',
    ],
)
