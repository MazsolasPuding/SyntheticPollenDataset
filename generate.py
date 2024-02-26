"""
Main Script for generating the Synthetic Data:
- Create Segmented Pollen Dataset - If Needed
- Split Dataset into Train, Validation, Test - If Needed
- Create Animation and save frames with annotations
"""

import argparse

import Animation
import Segmentation
import Tools

def main(
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        segment: bool,
        split: bool,
        config: bool,
        generate: bool):
    if segment:
        Segmentation.image_processing(
            path='/Users/horvada/Git/Personal/datasets/POLLEN73S',
            mode='all',
            num_samples=0,
            seed=42,
            preprocess=True,
            save_path='/Users/horvada/Git/Personal/datasets/POLLEN73S_SEG_BG/SegmentedPollens',
            keep_bg=True,
            plot_selection=False,
            plot_analytics=False
        )
    if split:
        Tools.data_split(
            input_path='/Users/horvada/Git/Personal/datasets/POLLEN73S_SEG_BG/SegmentedPollens',
            output_path='/Users/horvada/Git/Personal/datasets/POLLEN73S_SPLIT',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    if config:
        Tools.create_dataset_config(
            input_path='/Users/horvada/Git/Personal/datasets/POLLEN73S',
            output_path='/Users/horvada/Git/Personal/datasets/SYNTH_dataset_POLLEN73S'
        )
    if generate:
        for mode, ratio in {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}.items():
            if not ratio:
                continue
            Animation.create_synthetic_dataset(
                pollen_path='/Users/horvada/Git/Personal/datasets/POLLEN73S_SPLIT',
                output_path='/Users/horvada/Git/Personal/datasets/SYNTH_dataset_POLLEN73S',
                mode=mode,
                num_pollens=40,
                length=30,
                speed=10,
                fps=30,
                frame_size=(1920, 1080),
                save_video=True,
                save_frames=True,
                save_labels=True,
                draw_bb=True
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pollen_dataset", type=str, help="Input dataset Path", default="# /Users/horvada/Git/Personal/datasets/POLLEN73S")
    parser.add_argument("--train_ratio", type=float, default=0.78)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    parser.add_argument("--segment", type=bool, default=False)
    parser.add_argument("--split", type=bool, default=True)
    parser.add_argument("--config", type=bool, default=True)
    parser.add_argument("--generate", type=bool, default=True)
    args = parser.parse_args()

    main(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        segment=args.segment,
        split=args.split,
        config=args.config,
        generate=args.generate
    )

    # /Users/horvada/Git/Personal/datasets/POLLEN73S