from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bodies'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
print(os.path.join(os.path.dirname(__file__)))
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bullet_scene_interface', '../assets')
data_files = []

for root, dirs, files in os.walk(directory):
    for fn in files:
        data_files.append(os.path.join(root, fn))

setup(name='bullet_scene_interface',
      version='1.0',
      packages=find_packages(),
      python_requires='>=3',
      install_requires=['pybullet', 'numpy', 'scipy', 'numpngw'] + [
          'screeninfo==0.6.1' if sys.version_info >= (3, 6) else 'screeninfo==0.2'],
      long_description_content_type="text/markdown",
      author='Itamar Mishani',
      author_email="imishani@cmu.edu",
      package_data={'bullet_scene_interface': data_files})
