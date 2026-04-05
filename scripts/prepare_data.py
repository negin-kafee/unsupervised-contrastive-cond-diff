#!/usr/bin/env python3
"""
Data Preparation Script for Unsupervised Contrastive Conditional Diffusion Model

This script prepares the MOOD_IXI and BraTS datasets for training and testing.
It creates the necessary directory structure and CSV files.

Date: December 2024
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import nibabel as nib
from tqdm import tqdm
import argparse


def create_brain_mask(seg_path, output_path):
    """
    Create binary brain mask from tissue segmentation.
    The seg file has values 0,1,2,3 (background, CSF, GM, WM).
    Brain mask = any tissue > 0
    """
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    mask_data = (seg_data > 0).astype(np.float32)
    mask_img = nib.Nifti1Image(mask_data, seg_img.affine, seg_img.header)
    nib.save(mask_img, output_path)


def prepare_mood_ixi_data(
    source_dir: str,
    target_base_dir: str,
    n_folds: int = 5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Prepare MOOD_IXI dataset for training.
    
    Args:
        source_dir: Path to MOOD_IXI_all directory
        target_base_dir: Base directory for organized data
        n_folds: Number of cross-validation folds
        val_ratio: Ratio of validation data (from training)
        test_ratio: Ratio of test data
        seed: Random seed
    """
    print("=" * 60)
    print("Preparing MOOD_IXI Dataset")
    print("=" * 60)
    
    source_path = Path(source_dir)
    target_base = Path(target_base_dir)
    
    # Create target directories
    train_img_dir = target_base / "Train" / "mood_ixi" / "t2"
    train_mask_dir = target_base / "Train" / "mood_ixi" / "mask"
    splits_dir = target_base / "Data" / "splits"
    
    for d in [train_img_dir, train_mask_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all raw image files
    raw_dir = source_path / "input_noseg"
    seg_dir = source_path / "input_seg"
    
    raw_files = sorted([f for f in raw_dir.glob("*.nii.gz")])
    print(f"Found {len(raw_files)} raw images")
    
    # Create symlinks and brain masks
    all_subjects = []
    print("\nProcessing images and creating brain masks...")
    
    for raw_file in tqdm(raw_files):
        basename = raw_file.stem.replace('.nii', '')  # Remove .nii.gz
        
        # Try different seg file naming patterns
        seg_file = seg_dir / f"{basename}_seg.nii.gz"
        
        # Handle special naming for IXI T2 files
        if not seg_file.exists() and 'IXI' in basename:
            # Try alternative naming pattern for T2: IXI002-Guys-0828-T2_brain -> IXI002-Guys-0828_brain_seg_T2
            if '-T2_brain' in basename:
                alt_basename = basename.replace('-T2_brain', '_brain_seg_T2')
                seg_file = seg_dir / f"{alt_basename}.nii.gz"
            elif '-T1_brain' in basename:
                alt_basename = basename.replace('-T1_brain', '-T1_brain_seg')
                seg_file = seg_dir / f"{alt_basename}.nii.gz"
        
        if not seg_file.exists():
            print(f"Warning: No seg file for {basename}, skipping...")
            continue
        
        # Create symlink for image (treat as t2 for compatibility)
        img_name = f"{basename}_t2.nii.gz"
        mask_name = f"{basename}_mask.nii.gz"
        
        img_target = train_img_dir / img_name
        mask_target = train_mask_dir / mask_name
        
        # Create symlinks (to avoid copying large files)
        if not img_target.exists():
            os.symlink(raw_file, img_target)
        
        # Create brain mask from tissue segmentation
        if not mask_target.exists():
            create_brain_mask(str(seg_file), str(mask_target))
        
        all_subjects.append({
            'img_name': img_name,
            'age': 0,  # Unknown age
            'label': 0,  # Healthy subjects
            'img_path': f"/Train/mood_ixi/t2/{img_name}",
            'mask_path': f"/Train/mood_ixi/mask/{mask_name}",
            'seg_path': None
        })
    
    print(f"\nProcessed {len(all_subjects)} subjects")
    
    # Create DataFrame
    df = pd.DataFrame(all_subjects)
    
    # Split into train/val/test
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    
    n_test = int(len(df) * test_ratio)
    n_val_total = int(len(df) * val_ratio)
    
    test_indices = indices[:n_test]
    remaining_indices = indices[n_test:]
    
    # Save test split
    test_df = df.iloc[test_indices].reset_index(drop=True)
    test_df.to_csv(splits_dir / "MOOD_IXI_test.csv", index=True)
    print(f"Test set: {len(test_df)} subjects")
    
    # Create k-fold splits from remaining data
    train_val_df = df.iloc[remaining_indices].reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        train_fold_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        
        train_fold_df.to_csv(splits_dir / f"MOOD_IXI_train_fold{fold}.csv", index=True)
        val_fold_df.to_csv(splits_dir / f"MOOD_IXI_val_fold{fold}.csv", index=True)
        
        print(f"Fold {fold}: Train={len(train_fold_df)}, Val={len(val_fold_df)}")
    
    # Create avail_t2.csv (list of available t2 images)
    avail_t2 = pd.DataFrame({'0': df['img_name'].str.replace('_t2', '_t1')})
    avail_t2.to_csv(splits_dir / "avail_t2_mood.csv", index=False)
    
    print("\nMOOD_IXI preparation complete!")
    return df


def prepare_brats_data(
    source_dir: str,
    target_base_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    Prepare BraTS dataset for testing.
    
    Args:
        source_dir: Path to BraTS directory
        target_base_dir: Base directory for organized data
        val_ratio: Ratio of validation data
        seed: Random seed
    """
    print("\n" + "=" * 60)
    print("Preparing BraTS Dataset")
    print("=" * 60)
    
    source_path = Path(source_dir)
    target_base = Path(target_base_dir)
    
    # Create target directories
    test_img_dir = target_base / "Test" / "Brats20" / "t2"
    test_mask_dir = target_base / "Test" / "Brats20" / "mask"
    test_seg_dir = target_base / "Test" / "Brats20" / "seg"
    splits_dir = target_base / "Data" / "splits"
    
    for d in [test_img_dir, test_mask_dir, test_seg_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all subject directories
    brats_raw = source_path / "BraTS_raw"
    subject_dirs = sorted([d for d in brats_raw.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} BraTS subjects")
    
    all_subjects = []
    print("\nProcessing BraTS subjects...")
    
    for subj_dir in tqdm(subject_dirs):
        subj_name = subj_dir.name
        
        # Find T2 file
        t2_file = subj_dir / f"{subj_name}_t2.nii.gz"
        seg_file = subj_dir / f"{subj_name}_seg.nii.gz"
        
        if not t2_file.exists():
            print(f"Warning: No T2 file for {subj_name}, skipping...")
            continue
        
        if not seg_file.exists():
            print(f"Warning: No seg file for {subj_name}, skipping...")
            continue
        
        # Target filenames
        img_name = f"{subj_name}_t2.nii.gz"
        mask_name = f"{subj_name}_mask.nii.gz"
        seg_name = f"{subj_name}_seg.nii.gz"
        
        img_target = test_img_dir / img_name
        mask_target = test_mask_dir / mask_name
        seg_target = test_seg_dir / seg_name
        
        # Create symlinks
        if not img_target.exists():
            os.symlink(t2_file, img_target)
        
        # Create brain mask from T2 image (non-zero voxels)
        if not mask_target.exists():
            t2_img = nib.load(str(t2_file))
            t2_data = t2_img.get_fdata()
            mask_data = (t2_data > 0).astype(np.float32)
            mask_img = nib.Nifti1Image(mask_data, t2_img.affine, t2_img.header)
            nib.save(mask_img, str(mask_target))
        
        # Create symlink for segmentation (tumor labels)
        if not seg_target.exists():
            # Convert multi-class tumor seg to binary (any tumor > 0)
            seg_img = nib.load(str(seg_file))
            seg_data = seg_img.get_fdata()
            binary_seg = (seg_data > 0).astype(np.float32)
            binary_seg_img = nib.Nifti1Image(binary_seg, seg_img.affine, seg_img.header)
            nib.save(binary_seg_img, str(seg_target))
        
        all_subjects.append({
            'img_name': subj_name,
            'age': None,  # Unknown age
            'label': 1,  # Abnormal subjects (tumor)
            'img_path': f"/Test/Brats20/t2/{img_name}",
            'mask_path': f"/Test/Brats20/mask/{mask_name}",
            'seg_path': f"/Test/Brats20/seg/{seg_name}"
        })
    
    print(f"\nProcessed {len(all_subjects)} BraTS subjects")
    
    # Create DataFrame
    df = pd.DataFrame(all_subjects)
    
    # Split into val/test
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    n_val = int(len(df) * val_ratio)
    
    val_indices = indices[:n_val]
    test_indices = indices[n_val:]
    
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    val_df.to_csv(splits_dir / "Brats20_val.csv", index=True)
    test_df.to_csv(splits_dir / "Brats20_test.csv", index=True)
    
    print(f"Val set: {len(val_df)} subjects")
    print(f"Test set: {len(test_df)} subjects")
    print("\nBraTS preparation complete!")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for contrastive diffusion UAD')
    parser.add_argument('--mood_ixi_dir', type=str, 
                        default=None,
                        help='Path to MOOD_IXI_all directory')
    parser.add_argument('--brats_dir', type=str,
                        default=None,
                        help='Path to BraTS directory')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory for organized data')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Prepare MOOD_IXI data
    prepare_mood_ixi_data(
        source_dir=args.mood_ixi_dir,
        target_base_dir=args.output_dir,
        n_folds=args.n_folds,
        seed=args.seed
    )
    
    # Prepare BraTS data
    prepare_brats_data(
        source_dir=args.brats_dir,
        target_base_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Update pc_environment.env with DATA_DIR={args.output_dir}")
    print("2. Run training with: python run.py experiment=contrastive_encoder")


if __name__ == '__main__':
    main()
