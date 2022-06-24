## Architecture 

For reference, we print all learnable layers in our architecture and their detailed parameters using torch-summary.  We set the default batch size to be 8. The size of each input point cloud is set as 256*6 (3 for positional information, 3 for extra features). 



```
------------------------------------------------------------------------------------------------------------------
  Layer (type)                                                Input Shape         Param #     Tr. Param #
==================================================================================================================
MultiScaleEncoder-1                                [8, 3, 256], [8, 3, 256]          64,768          64,768
FeatureCorrelator-2  [8, 3, 256], [8, 3, 256], [8, 512, 256], [8, 512, 256]       1,063,184       1,063,184
FlowDecoder-3        [8, 3, 256], [8, 3, 256], [8, 512, 256], [8, 512, 256]       2,929,728       2,929,728
==================================================================================================================
Total params: 4,057,680
Trainable params: 4,057,680
Non-trainable params: 0
------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------
  Layer (type)        Output Shape         Param #     Tr. Param #
===========================================================================
MultiScaleEncoder-1       [8, 256, 256]     64,768          64,768
FeatureCorrelator-2       [8, 512, 256]     1,063,184       1,063,184
FlowDecoder-3             [8, 3, 256]       2,929,728       2,929,728
===========================================================================
Total params: 4,057,680
Trainable params: 4,057,680
Non-trainable params: 0
---------------------------------------------------------------------------


============================================= Hierarchical Summary =============================================

RaFlow(
(mse_layer): MultiScaleEncoder(
(ms_ls): ModuleList(
(0): PointLocalFeature(
(mlp_convs): ModuleList(
  (0): Conv2d(6, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 192 params
  (1): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,024 params
  (2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 2,048 params
), 3,264 params
(mlp_bns): ModuleList(
  (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 256 params
(mlp2_convs): ModuleList(
  (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
), 12,288 params
(mlp2_bns): ModuleList(
  (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 384 params
(queryandgroup): QueryAndGroup(), 0 params
), 16,192 params
(1): PointLocalFeature(
(mlp_convs): ModuleList(
  (0): Conv2d(6, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 192 params
  (1): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,024 params
  (2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 2,048 params
), 3,264 params
(mlp_bns): ModuleList(
  (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 256 params
(mlp2_convs): ModuleList(
  (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
), 12,288 params
(mlp2_bns): ModuleList(
  (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 384 params
(queryandgroup): QueryAndGroup(), 0 params
), 16,192 params
(2): PointLocalFeature(
(mlp_convs): ModuleList(
  (0): Conv2d(6, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 192 params
  (1): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,024 params
  (2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 2,048 params
), 3,264 params
(mlp_bns): ModuleList(
  (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 256 params
(mlp2_convs): ModuleList(
  (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
), 12,288 params
(mlp2_bns): ModuleList(
  (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 384 params
(queryandgroup): QueryAndGroup(), 0 params
), 16,192 params
(3): PointLocalFeature(
(mlp_convs): ModuleList(
  (0): Conv2d(6, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 192 params
  (1): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,024 params
  (2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 2,048 params
), 3,264 params
(mlp_bns): ModuleList(
  (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 64 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 256 params
(mlp2_convs): ModuleList(
  (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
), 12,288 params
(mlp2_bns): ModuleList(
  (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
), 384 params
(queryandgroup): QueryAndGroup(), 0 params
), 16,192 params
), 64,768 params
), 64,768 params
(fc_layer): FeatureCorrelator(
(mlp_convs): ModuleList(
(0): Conv2d(1027, 512, kernel_size=(1, 1), stride=(1, 1)), 526,336 params
(1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)), 262,656 params
(2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)), 262,656 params
), 1,051,648 params
(weightnet1): WeightNet(
(mlp_convs): ModuleList(
(0): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1)), 32 params
(1): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1)), 72 params
(2): Conv2d(8, 512, kernel_size=(1, 1), stride=(1, 1)), 4,608 params
), 4,712 params
(mlp_bns): ModuleList(
(0): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 16 params
(1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 16 params
(2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 1,024 params
), 1,056 params
), 5,768 params
(weightnet2): WeightNet(
(mlp_convs): ModuleList(
(0): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1)), 32 params
(1): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1)), 72 params
(2): Conv2d(8, 512, kernel_size=(1, 1), stride=(1, 1)), 4,608 params
), 4,712 params
(mlp_bns): ModuleList(
(0): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 16 params
(1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 16 params
(2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 1,024 params
), 1,056 params
), 5,768 params
(relu): LeakyReLU(negative_slope=0.1, inplace=True), 0 params
), 1,063,184 params
(fd_layer): FlowDecoder(
(mse): MultiScaleEncoder(
(ms_ls): ModuleList(
(0): PointLocalFeature(
  (mlp_convs): ModuleList(
    (0): Conv2d(1030, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 527,360 params
    (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
    (2): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
  ), 674,816 params
  (mlp_bns): ModuleList(
    (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 1,024 params
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 1,664 params
  (mlp2_convs): ModuleList(
    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  ), 12,288 params
  (mlp2_bns): ModuleList(
    (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 384 params
  (queryandgroup): QueryAndGroup(), 0 params
), 689,152 params
(1): PointLocalFeature(
  (mlp_convs): ModuleList(
    (0): Conv2d(1030, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 527,360 params
    (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
    (2): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
  ), 674,816 params
  (mlp_bns): ModuleList(
    (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 1,024 params
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 1,664 params
  (mlp2_convs): ModuleList(
    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  ), 12,288 params
  (mlp2_bns): ModuleList(
    (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 384 params
  (queryandgroup): QueryAndGroup(), 0 params
), 689,152 params
(2): PointLocalFeature(
  (mlp_convs): ModuleList(
    (0): Conv2d(1030, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 527,360 params
    (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
    (2): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
  ), 674,816 params
  (mlp_bns): ModuleList(
    (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 1,024 params
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 1,664 params
  (mlp2_convs): ModuleList(
    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  ), 12,288 params
  (mlp2_bns): ModuleList(
    (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 384 params
  (queryandgroup): QueryAndGroup(), 0 params
), 689,152 params
(3): PointLocalFeature(
  (mlp_convs): ModuleList(
    (0): Conv2d(1030, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 527,360 params
    (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
    (2): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
  ), 674,816 params
  (mlp_bns): ModuleList(
    (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 1,024 params
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 1,664 params
  (mlp2_convs): ModuleList(
    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
    (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
  ), 12,288 params
  (mlp2_bns): ModuleList(
    (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  ), 384 params
  (queryandgroup): QueryAndGroup(), 0 params
), 689,152 params
), 2,756,608 params
), 2,756,608 params
(fp): FlowPredictor(
(sf_mlp): ModuleList(
(0): Sequential(
  (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
  (2): ReLU(), 0 params
), 131,584 params
(1): Sequential(
  (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), 32,768 params
  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 256 params
  (2): ReLU(), 0 params
), 33,024 params
(2): Sequential(
  (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 8,192 params
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 128 params
  (2): ReLU(), 0 params
), 8,320 params
), 172,928 params
(conv2): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), 192 params
), 173,120 params
), 2,929,728 params
), 4,057,680 params

```

