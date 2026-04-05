from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Optional
import pandas as pd
import src.datamodules.create_dataset as create_dataset
from src.datamodules.create_dataset import NOVADataset


class Brats21(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(Brats21, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats21.IDs.val
        self.csvpath_test = cfg.path.Brats21.IDs.test
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats21'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1',cfg.mode).str.replace('FLAIR.nii.gz',f'{cfg.mode.lower()}.nii.gz')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20_T1_seg(LightningDataModule):
    """
    DataModule for BraTS 2020 dataset evaluation using FSL FAST segmented T1-weighted images.
    """

    def __init__(self, cfg, fold=None):
        super(Brats20_T1_seg, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats20_T1_seg.IDs.val
        self.csvpath_test = cfg.path.Brats20_T1_seg.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats20_T1_seg'
            
            # Only prepend pathBase if paths are not already absolute
            for col in ['img_path', 'mask_path', 'seg_path']:
                if not self.csv[state][col].iloc[0].startswith('/'):
                    self.csv[state][col] = cfg.path.pathBase + '/' + self.csv[state][col]

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20_T2_seg(LightningDataModule):
    """
    DataModule for BraTS 2020 dataset evaluation using FSL FAST segmented T2-weighted images.
    """

    def __init__(self, cfg, fold=None):
        super(Brats20_T2_seg, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats20_T2_seg.IDs.val
        self.csvpath_test = cfg.path.Brats20_T2_seg.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats20_T2_seg'
            
            # Only prepend pathBase if paths are not already absolute
            for col in ['img_path', 'mask_path', 'seg_path']:
                if not self.csv[state][col].iloc[0].startswith('/'):
                    self.csv[state][col] = cfg.path.pathBase + '/' + self.csv[state][col]

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20_T1_H5_seg(LightningDataModule):
    """
    DataModule for BraTS 2020 dataset evaluation using FSL FAST segmented T1-weighted images from H5 files.
    H5 files contain proper 4-class tissue segmentation (0=background, 1=CSF, 2=GM, 3=WM).
    """

    def __init__(self, cfg, fold=None):
        super(Brats20_T1_H5_seg, self).__init__()
        self.cfg = cfg
        self.h5_img_path = cfg.path.Brats20_T1_H5_seg.h5_img
        self.h5_seg_path = cfg.path.Brats20_T1_H5_seg.h5_seg

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            # Use same data for val and test (BraTS has no separate val/test split in H5)
            if self.cfg.sample_set:
                # For sample_set, we'll just load a subset by modifying the dataset after creation
                self.val_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='val', setname='Brats20_T1_H5_seg'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='test', setname='Brats20_T1_H5_seg'
                )
            else:
                self.val_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='val', setname='Brats20_T1_H5_seg'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='test', setname='Brats20_T1_H5_seg'
                )

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20_T2_H5_seg(LightningDataModule):
    """
    DataModule for BraTS 2020 dataset evaluation using FSL FAST segmented T2-weighted images from H5 files.
    H5 files contain proper 4-class tissue segmentation (0=background, 1=CSF, 2=GM, 3=WM).
    """

    def __init__(self, cfg, fold=None):
        super(Brats20_T2_H5_seg, self).__init__()
        self.cfg = cfg
        self.h5_img_path = cfg.path.Brats20_T2_H5_seg.h5_img
        self.h5_seg_path = cfg.path.Brats20_T2_H5_seg.h5_seg

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='val', setname='Brats20_T2_H5_seg'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='test', setname='Brats20_T2_H5_seg'
                )
            else:
                self.val_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='val', setname='Brats20_T2_H5_seg'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.h5_img_path, self.h5_seg_path, self.cfg,
                    settype='test', setname='Brats20_T2_H5_seg'
                )

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MSLUB(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(MSLUB, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.MSLUB.IDs.val
        self.csvpath_test = cfg.path.MSLUB.IDs.test
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MSLUB'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']
            
            if cfg.mode != 't1':
                #self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('uniso/t1',f'uniso/{cfg.mode}').str.replace('t1.nii.gz',f'{cfg.mode}.nii.gz')
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('MSLUB/t1',f'MSLUB/{cfg.mode}').str.replace('t1.nii.gz',f'{cfg.mode}.nii.gz')
    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:4], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:4], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20(LightningDataModule):
    """
    DataModule for BraTS 2020 dataset evaluation.
    Uses the custom BraTS dataset structure.
    """

    def __init__(self, cfg, fold=None):
        super(Brats20, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats20.IDs.val
        self.csvpath_test = cfg.path.Brats20.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats20'
            
            self.csv[state]['img_path'] = cfg.path.pathBase + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


# NOTE: Added here to handle NOVA dataset
class NOVA(LightningDataModule):
    """
    Class that handle the loading of NOVA dataset.
    """
    def __init__(self, cfg, fold=None):
        super(NOVA, self).__init__()
        self.cfg = cfg
        self.csv_path = cfg.path.NOVA.csv_path
        self.image_dir = cfg.path.NOVA.image_dir

    def setup(self, stage: Optional[str] = None):
        self.test_dataset = NOVADataset(config=self.cfg, csv_path=self.csv_path, image_dir=self.image_dir)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20_T1(LightningDataModule):
    """
    DataModule for BraTS 2020 dataset evaluation using T1-weighted images.
    Uses separate T1-specific CSVs with paths already pointing to T1 data.
    """

    def __init__(self, cfg, fold=None):
        super(Brats20_T1, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats20_T1.IDs.val
        self.csvpath_test = cfg.path.Brats20_T1.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats20_T1'
            
            # Only prepend pathBase if paths are not already absolute
            for col in ['img_path', 'mask_path', 'seg_path']:
                if not self.csv[state][col].iloc[0].startswith('/'):
                    self.csv[state][col] = cfg.path.pathBase + '/' + self.csv[state][col]

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20_T2(LightningDataModule):
    """
    DataModule for BraTS 2020 dataset evaluation using T2-weighted images.
    Uses separate T2-specific CSVs with paths already pointing to T2 data.
    """

    def __init__(self, cfg, fold=None):
        super(Brats20_T2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats20_T2.IDs.val
        self.csvpath_test = cfg.path.Brats20_T2.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats20_T2'
            
            # Only prepend pathBase if paths are not already absolute
            for col in ['img_path', 'mask_path', 'seg_path']:
                if not self.csv[state][col].iloc[0].startswith('/'):
                    self.csv[state][col] = cfg.path.pathBase + '/' + self.csv[state][col]

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


