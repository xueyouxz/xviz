#!/usr/bin/env python3
import yaml
from typing import Dict, List, Any

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
from work.convert_v3.converters import NuScenesConverter

LOCATIONS = [
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
    "boston-seaport",
]


def load_config(config_path: str) -> Dict[str, Any]:
    print(f"加载配置文件：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_scene_names(config: Dict[str, Any]) -> List[str]:
    from nuscenes.utils import splits

    scenes_config = config['scenes']
    if scenes_config == 'mini_val':
        return splits.mini_val
    elif scenes_config == 'mini_train':
        return splits.mini_train
    elif scenes_config == 'train':
        return splits.train
    elif scenes_config == 'val':
        return splits.val
    else:
        return [scenes_config]


def get_map_name(nuscenes, scene_name):
    scenes = [s for s in nuscenes.scene if s['name'] == scene_name]
    if not scenes:
        raise ValueError(f"未找到场景：{scene_name}")
    scene = scenes[0]
    return nuscenes.get("log", scene["log_token"])["location"]


def batch_convert_scenes(config_path: str):
    config = load_config(config_path)

    scene_names = get_scene_names(config)

    print(f"将转换 {len(scene_names)} 个场景")
    map_explorers = {}
    nusc = NuScenes(
        version=config['version'],
        dataroot=config['data_root'],
        verbose=True
    )
    for map_name in LOCATIONS:
        nusc_map = NuScenesMap(dataroot=config['data_root'], map_name=map_name)
        map_explorers[map_name] = NuScenesMapExplorer(nusc_map)


    for index, scene_name in enumerate(scene_names):
        scene_map_name = get_map_name(nusc, scene_name)
        converter = NuScenesConverter(
            nuscenes_root=config['data_root'],
            output_dir=config['output_dir'],
            version=config['version'],
            image_max_width=config.get('image_max_width', 400),
            image_max_height=config.get('image_max_height', 300),
            nuscenes_obj=nusc,
            map_explorer=map_explorers[scene_map_name]
        )

        converter.convert_scene(index=index,
                                scene_name=scene_name,
                                output_name=scene_name,
                                sample_limit=config.get('sample_limit', None)
                                )


if __name__ == '__main__':
    batch_convert_scenes('config.yaml')
