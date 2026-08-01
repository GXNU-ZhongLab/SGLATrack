from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/public/zhongbineng/datasets/got10k_lmdb'
    settings.got10k_path = '/public/zhongbineng/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/public/zhongbineng/datasets/itb'
    settings.lasot_extension_subset_path_path = '/public/zhongbineng/datasets/lasot_extension_subset'
    settings.lasot_lmdb_path = '/public/zhongbineng/datasets/lasot_lmdb'
    settings.lasot_path = '/public/zhongbineng/datasets/lasot'
    settings.network_path = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/public/zhongbineng/datasets/nfs'
    settings.otb_path = '/public/zhongbineng/datasets/otb'
    settings.prj_dir = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s'
    settings.result_plot_path = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s/output/test/result_plots'
    settings.results_path = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s/output'
    settings.segmentation_path = '/public/zhongbineng/workspaces/xcc25/TPAMI/ASTrack_t6i6_caev2S_MLP5_4s/output/test/segmentation_results'
    settings.tc128_path = '/public/zhongbineng/datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/public/zhongbineng/datasets/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/public/zhongbineng/datasets/trackingnet'
    settings.uav_path = '/public/zhongbineng/datasets/uav'
    settings.vot18_path = '/public/zhongbineng/datasets/vot2018'
    settings.vot22_path = '/public/zhongbineng/datasets/vot2022'
    settings.vot_path = '/public/zhongbineng/datasets/VOT2019'
    settings.youtubevos_dir = ''
    settings.uavdt_path = '/public/zhongbineng/datasets/AAAXCC/uavdt'
    settings.dtb70_path = '/public/zhongbineng/datasets/AAAXCC/DTB70'
    settings.visdrone_path = '/public/zhongbineng/datasets/AAAXCC/VisDrone2018-SOT-test-dev'
    settings.uavtrack_path = '/public/zhongbineng/datasets/AAAXCC/V4RFlight112'
    settings.uav123_path = '/public/zhongbineng/datasets/AAAXCC/UAV123'
    settings.uav123_10fps_path = '/public/zhongbineng/datasets/AAAXCC/UAV123_10fps'
    settings.webuav3m_path = '/public/zhongbineng/datasets/AAAXCC/WebUAV-3M/Test'
    return settings
