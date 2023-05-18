_base_ = ['_base_/default_runtime.py']

model = dict(type='BPResNet1D',
             data_preprocessor=dict(type='RppgDataPreprocessor'),
             backbone=dict(
                 type='ResNet1D',
                 depth=50,
                 frozen_stages=4,
                 init_cfg=dict(
                     type='Pretrained',
                     checkpoint='data/resnet_ppg_nonmixed_backbone_v2.pb')),
             neck=dict(type='AveragePooling'),
             head=dict(type='BPDenseHead', loss=dict(type='MSELoss')))

dataset_type = 'RppgData'
train_pipeline = [
    dict(type='PackInputs', input_key='rppg'),
]
test_pipeline = [
    dict(type='PackInputs', input_key='rppg'),
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(type=dataset_type,
                 ann_file='data/rPPG-BP-UKL_rppg_7s.h5',
                 used_subjects=[7.,  8., 10., 13., 14., 19., 21., 23., 24., 27., 33., 35., 36., 44., 46., 48.],
                 data_prefix='',
                 test_mode=False,
                 pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(type=dataset_type,
                 ann_file='data/rPPG-BP-UKL_rppg_7s.h5',
                 used_subjects=[6],
                 data_prefix='',
                 test_mode=True,
                 pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='MAE', gt_key='gt_label', pred_key='pred_label')
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam',
                                    lr=0.001,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=0,
                                    amsgrad=False))

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
)
