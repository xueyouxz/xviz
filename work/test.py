import yaml
from nuscenes.nuscenes import NuScenes

config_path = 'converter_v1/config.yaml'
config = yaml.load(open(config_path,encoding='utf-8'), Loader=yaml.FullLoader)
nuscenes = NuScenes(version=config['version'], dataroot=config['data_root'], verbose=True)
print('nuscenes debug')