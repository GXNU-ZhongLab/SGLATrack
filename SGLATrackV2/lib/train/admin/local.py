class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s/pretrained_networks'
        self.lasot_dir = '/public/zhongbineng/datasets/lasot'
        self.got10k_dir = '/public/zhongbineng/datasets/got10k/train'
        self.got10k_val_dir = '/public/zhongbineng/datasets/got10k/val'
        self.lasot_lmdb_dir = '/public/zhongbineng/datasets/lasot_lmdb'
        self.got10k_lmdb_dir = '/public/zhongbineng/datasets/got10k_lmdb'
        self.trackingnet_dir = '/public/zhongbineng/datasets/trackingnet'
        self.trackingnet_lmdb_dir = '/public/zhongbineng/datasets/trackingnet_lmdb'
        self.coco_dir = '/public/zhongbineng/datasets/coco'
        self.coco_lmdb_dir = '/public/zhongbineng/datasets/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/public/zhongbineng/datasets/vid'
        self.imagenet_lmdb_dir = '/public/zhongbineng/datasets/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
