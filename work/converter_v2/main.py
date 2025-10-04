import os

from nuscenes import NuScenes
from nuscenes.utils import splits
import yaml

from work.converter_v2.nuscenes_converter import NuscenesConverter
from xviz_avs.io import DirectorySource, XVIZGLBWriter


def batch_convert_scenes(config_path):
    """批量转换场景"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 初始化NuScenes
    nusc = NuScenes(
        version=config['version'],
        dataroot=config['data_root'],
        verbose=True
    )

    # 获取场景列表
    if config['scenes'] == 'mini_val':
        scene_names = splits.mini_val
    elif config['scenes'] == 'mini_train':
        scene_names = splits.mini_train
    else:
        scene_names = [config['scenes']]

    # 逐个转换
    for scene_name in scene_names:

        output_path = os.path.join(config['output_dir'], scene_name)
        os.makedirs(output_path, exist_ok=True)

        print(f'\n{"=" * 60}')
        print(f'Converting scene: {scene_name}')
        print(f'{"=" * 60}')

        converter = NuscenesConverter(
            scene_name=scene_name,
            output_path=output_path,
            nuscenes=nusc,
            config=config
        )

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

            # 进度显示
            if (i + 1) % 10 == 0:
                print(f'Progress: {i + 1}/{sample_limit} frames')

        writer.close()
        print(f'✓ Completed: {scene_name}')


# 使用
batch_convert_scenes('config.yaml')
