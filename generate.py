"""
Main Wrapper Script for generating the Synthetic Dataset:
- Create Segmented Pollen Dataset
- Split Dataset into Train, Validation, Test
- Create Animation and save frames with annotations
- Create Dataset Config Yaml for training and for reproducible generation
"""

import yaml
from pathlib import Path
from pprint import pprint
import shutil

import Animation
import Segmentation
import Tools

def main(config):
    pprint(config)
    if config['flags']['segment']:
        Segmentation.image_processing(
            path=config['path']['pollen_dataset'],
            mode=config['segment']['mode'],
            num_samples=config['segment']['num_samples'],
            seed=config['segment']['seed'],
            preprocess=config['segment']['preprocess'],
            save_path=config['segment']['save_path'],
            keep_bg=config['segment']['keep_bg'],
            plot_selection=config['segment']['plot_selection'],
            plot_analytics=config['segment']['plot_analytics']
        )
    if config['flags']['split']:
        Tools.data_split(
            input_path=config['path']['segmented_path'],
            output_path=config['path']['split_segmented_path'],
            split_mode=config['split']['split_mode'],
            train_ratio=config['split']['train_ratio'],
            val_ratio=config['split']['val_ratio'],
            test_ratio=config['split']['test_ratio']
        )
    if config['flags']['generate']:
        outpath = Path(config['path']['synth_dataset_path'])
        if outpath.exists():
            shutil.rmtree(outpath)

        for mode, pars in {
            'train': [config['split']['train_ratio'], config['synthetic']['train_length']],
            'val': [config['split']['val_ratio'], config['synthetic']['val_length']],
            'test': [config['split']['test_ratio'], config['synthetic']['test_length']]
        }.items():
            if not pars[0]:
                continue
            sdc = Animation.SyntheticDatasetCreator(
                pollen_path=config['path']['split_segmented_path'],
                background_path=Path(config['path']['segmented_path']) / 'SegmentedBackground',
                output_path=config['path']['synth_dataset_path'],
                mode=mode,
                pollen_pos_mode=config['synthetic']['pollen_pos_mode'],
                num_pollens=config['synthetic']['num_pollens'],
                pollen_to_frame_ratio=config['synthetic']['pollen_to_frame_ratio'],
                augment=config['synthetic']['augment'],
                background_type=config['synthetic']['background_type'],
                background_regen_inteerval=config['synthetic']['background_regen_inteerval'],
                background_movement=config['synthetic']['background_movement'],
                length=pars[1],
                speed=config['synthetic']['speed'],
                fps=config['synthetic']['fps'],
                frame_size=tuple(config['synthetic']['frame_size']),
                save_video=config['synthetic']['save_video'],
                save_frames=config['synthetic']['save_frames'],
                save_labels=config['synthetic']['save_labels'],
                draw_bb=config['synthetic']['draw_bb'],
            )
            sdc()
            with open(config['path']['synth_dataset_path'] + "/generate.yaml", 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
    if config['flags']['train_config']:
        Tools.create_dataset_config(
            input_path=config['path']['pollen_dataset'],
            output_path=config['path']['synth_dataset_path']
        )
    print(f"Generated dataset!")



if __name__ == "__main__":
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
   