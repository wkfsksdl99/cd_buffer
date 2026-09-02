import torchvision
from torchvision.models.detection.rpn import AnchorGenerator


SUPPORTED_BACKBONES = {'resnet18', 'resnet50', 'resnet101'}


def create_model_improved(
    num_classes,
    backbone_type='resnet50',
    pretrained=True,
    min_size=352,
    max_size=1216,
    pretrained_path=None,
):
    if backbone_type not in SUPPORTED_BACKBONES:
        supported = ', '.join(sorted(SUPPORTED_BACKBONES))
        raise ValueError(
            f'Unknown backbone type: {backbone_type}. Supported types: {supported}'
        )

    weights = (
        torchvision.models.get_model_weights(backbone_type).DEFAULT
        if pretrained
        else None
    )
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name=backbone_type,
        weights=weights,
    )
    model = torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
    )
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    return model

