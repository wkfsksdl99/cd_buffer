import argparse


class CDArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, light=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.light = light

    def parse_args(self, args=None, namespace=None):
        parsed_args = super().parse_args(args=args, namespace=namespace)
        if parsed_args.save_dir is None:
            result_root = (
                './cd_buffer_tta_results_light'
                if self.light
                else './cd_buffer_tta_results'
            )
            parsed_args.save_dir = (
                f'{result_root}/{parsed_args.dataset}_'
                f'{parsed_args.target_task}_{parsed_args.target_severity}'
            )
        return parsed_args


def build_parser(light=False):
    parser = CDArgumentParser(description='cd_buffer tta', light=light)
    dataset_choices = (
        ['kitti', 'cityscapes']
        if light
        else ['kitti', 'cityscapes', 'acdc']
    )
    if light:
        dataset_default = 'cityscapes'
        target_split = 'val'
        target_task = 'foggy'
        target_severity = '0.01'
        weights = './models/cityscapes/best_model.pth'
        device = 'cuda:0'
        dataroot = './dataset/cityscapes'
        source_stats_path = './source_stats_cityscapes.pth'
        ema_gamma = 256
    else:
        dataset_default = 'cityscapes'
        target_split = 'val'
        target_task = 'foggy'
        target_severity = '0.01'
        weights = './models/cityscapes/best_model.pth'
        device = 'cuda:0'
        dataroot = './dataset/cityscapes'
        source_stats_path = './source_stats_cityscapes.pth'
        ema_gamma = 256
    parser.add_argument('--dataset', default=dataset_default, choices=dataset_choices)
    parser.add_argument('--source_split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--target_split', default=target_split, choices=['train', 'val', 'test'])
    parser.add_argument('--target_task', default=target_task)
    parser.add_argument('--target_severity', default=target_severity)
    parser.add_argument('--weights', default=weights)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default=device)
    parser.add_argument('--dataroot', default=dataroot)
    parser.add_argument('--acdc_dataroot', default='./dataset/acdc')
    if not light:
        parser.add_argument('--lambda_wreg', type=float, default=0.05)
        parser.add_argument('--tta_lr', type=float, default=0.0001)
        parser.add_argument('--bn_lr', type=float, default=0.0001)
    parser.add_argument('--source_stats_path', default=source_stats_path)
    parser.add_argument('--alpha_gl', type=float, default=1.0)
    parser.add_argument('--alpha_fg', type=float, default=1.0)
    parser.add_argument('--gl_align', default='KL')
    parser.add_argument('--fg_align', default='KL')
    parser.add_argument('--ema_gamma', type=int, default=ema_gamma)
    parser.add_argument('--fg_conf_thresh', type=float, default=0.5)
    parser.add_argument('--num_stats_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pruning_ratio_start', type=float, default=0.02)
    parser.add_argument('--pruning_ratio_end', type=float, default=0.05)
    parser.add_argument('--pruning_ratio_schedule', default='constant', choices=['constant', 'linear'])
    parser.add_argument('--pruning_threshold_mode', default='global', choices=['global', 'layer'])
    parser.add_argument('--pruning_threshold_epsilon', type=float, default=1e-06)
    parser.add_argument('--pruning_reactivation_prob', type=float, default=0.05)
    return parser
