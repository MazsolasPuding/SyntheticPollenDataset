"""
Main Wrapper Script for generating the Synthetic Dataset:
- Create Segmented Pollen Dataset
- Split Dataset into Train, Validation, Test
- Create Dataset Config Yaml
- Create Animation and save frames with annotations
"""

import argparse

import Animation
import Segmentation
import Tools

def main(
        pollen_dataset: str,
        segmented_path: str,
        split_segmented_path: str,
        synth_dataset_path: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        train_length: int,
        val_length: int,
        test_length: int,
        segment: bool,
        split: bool,
        config: bool,
        generate: bool):
    if segment:
        Segmentation.image_processing(
            path=pollen_dataset,
            mode='all',
            num_samples=0,
            seed=42,
            preprocess=True,
            save_path=segmented_path,
            keep_bg=True,
            plot_selection=False,
            plot_analytics=False
        )
    if split:
        Tools.data_split(
            input_path=segmented_path,
            output_path=split_segmented_path,
            split_mode ='SegmentedPollens',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    if config:
        Tools.create_dataset_config(
            input_path=pollen_dataset,
            output_path=synth_dataset_path
        )
    if generate:
        for mode, pars in {'train': [train_ratio, train_length], 'val': [val_ratio, val_length], 'test': [test_ratio, test_length]}.items():
            if not pars[0]:
                continue
            Animation.create_synthetic_dataset(
                pollen_path=split_segmented_path,
                output_path=synth_dataset_path,
                mode=mode,
                pollen_pos_mode = "random",
                num_pollens=20,
                length=pars[1],
                speed=5,
                fps=30,
                frame_size=(1920, 1080),
                save_video=True,
                save_frames=True,
                save_labels=True,
                draw_bb=False
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Windows Paths
    # parser.add_argument("--pollen_dataset", type=str, default="E:/coding/Pollen/datasets/POLLEN73S")
    # parser.add_argument("--segmented_path", type=str, default="E:/coding/Pollen/datasets/POLLEN73S_SEG_BG")
    # parser.add_argument("--split_segmented_path", type=str, default="E:/coding/Pollen/datasets/POLLEN73S_SEG_SPLIT_80TRAIN_20VAL") # 80% training split 300 seconds video length
    # parser.add_argument("--synth_dataset_path", type=str, default="E:/coding/Pollen/datasets/SYNTH_POLLEN73S_300_60_TEST")
    # parser.add_argument("--pollen_dataset", type=str, default="E:/coding/Pollen/datasets/POLEN23E_Structured")
    # parser.add_argument("--segmented_path", type=str, default="E:/coding/Pollen/datasets/POLEN23E_SEG_BG_Manual_Filtered")
    # parser.add_argument("--split_segmented_path", type=str, default="E:/coding/Pollen/datasets/POLEN23E_SEG_SPLIT_Manual_Filtered_80TRAIN_20VAL") # 80% training split 300 seconds video length
    # parser.add_argument("--synth_dataset_path", type=str, default="E:/coding/Pollen/datasets/SYNTH_POLEN23E_Manual_Filtered_300_60")
    # Mac Paths
    parser.add_argument("--pollen_dataset", type=str, default="/Users/horvada/Git/Personal/datasets/POLLEN73S")
    parser.add_argument("--segmented_path", type=str, default="/Users/horvada/Git/Personal/datasets/POLLEN73S_SEG_BG")
    parser.add_argument("--split_segmented_path", type=str, default="/Users/horvada/Git/Personal/datasets/POLLEN73S_SEG_SPLIT_TEST")
    parser.add_argument("--synth_dataset_path", type=str, default='/Users/horvada/Git/Personal/datasets/SYNTH_POLLEN73S_TEST')

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.0)

    parser.add_argument("--train_length", type=int, default=10)
    parser.add_argument("--val_length", type=int, default=6)
    parser.add_argument("--test_length", type=int, default=0)

    parser.add_argument("--segment", type=bool, default=False)
    parser.add_argument("--split", type=bool, default=True)
    parser.add_argument("--config", type=bool, default=True)
    parser.add_argument("--generate", type=bool, default=True)
    args = parser.parse_args()

    main(
        pollen_dataset=args.pollen_dataset,
        segmented_path=args.segmented_path,
        split_segmented_path=args.split_segmented_path,
        synth_dataset_path=args.synth_dataset_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_length=args.train_length,
        val_length=args.val_length,
        test_length=args.test_length,
        segment=args.segment,
        split=args.split,
        config=args.config,
        generate=args.generate
    )
