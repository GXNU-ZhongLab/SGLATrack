import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


datasets = ['uav123','uav123_10fps','uavdt','dtb70','visdrone','uavtrack','uavtrack112']
# datasets = ['uav20l']
run_ids_list = [ 300]
for dataset_name in datasets:
    # 获取当前数据集
    dataset = get_dataset(dataset_name)
    
    # 为当前数据集创建trackers列表
    trackers = []
    for run_id in run_ids_list:
        # 添加不同run_id的跟踪器，使用display_name区分
        tt = trackerlist(
            name='sglatrackv2',
            parameter_name='baseline_WOCE_cae',
            dataset_name=dataset_name,
            run_ids=[run_id],  # 注意这里需要列表类型
            display_name=f'sglatrackv2_run{run_id}'  # 添加run_id后缀以便区分
        )
        # 打印当前数据集的结果
        # print(f"\nResults for dataset: {dataset_name}")
        print_results(tt, dataset, dataset_name, merge_results=True, 
                    plot_types=('success', 'norm_prec', 'prec'))
