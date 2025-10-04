import os

from nuscenes.nuscenes import NuScenes
import yaml
from nuscenes.utils import splits

from work.converter_v1.converter import NuscenesConverter
from xviz_avs.io import DirectorySource, XVIZGLBWriter


def get_scene_names(data_type):
    if data_type == 'train':
        return splits.train
    elif data_type == 'val':
        return splits.val
    elif data_type == 'test':
        return splits.test
    elif data_type == 'mini_train':
        return splits.mini_train
    elif data_type == 'mini_val':
        return splits.mini_val
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def transform(config):
    nuscenes = NuScenes(version=config['version'], dataroot=config['data_root'], verbose=True)
    scene_names = get_scene_names(config['scenes'])
    for scene_name in scene_names:
        output_path = os.path.join(config['output_dir'], scene_name)
        converter = NuscenesConverter(scene_name=scene_name, output_path=output_path, nuscenes=nuscenes, config=config)
        print(f'Converting {scene_name}')
        converter.initialize()

        sink = DirectorySource(output_path)
        writer = XVIZGLBWriter(sink)

        metadata = converter.get_metadata()
        writer.write_message(metadata)
        sample_limit = min(config['sample_limit'], converter.sample_count)
        for i in range(sample_limit):
            xviz_message = converter.convert_message(i)
            if xviz_message is not None:
                writer.write_message(xviz_message)
        writer.close()
        print(f'Finished {scene_name}')


if __name__ == '__main__':
    config_path = 'config.yaml'
    config = yaml.load(open(config_path,encoding='utf-8'), Loader=yaml.FullLoader)
    transform(config)
    # todo: 使用pyyaml加载配置文件\并给出使用示例
