from mmdet.apis import DetInferencer

# 初始化推理器
inferencer = DetInferencer(
    model='/root/mmdetection/configs/ssd/ssd300_evd.py',
    weights='/root/mmdetection/work_dirs/ssd300_evd/epoch_24.pth',
    device='cuda:0'
)

# 推理单张图片
result = inferencer(
    '/root/a-photo-of-a-road-with-various-vehicles-_ByNDCdV0Tt6CuMGGc8epGA_y6UTIHChTXmHhtsu6fM8bA.jpeg',
    out_dir='/root/mmdetection/vis/ssd result',
    pred_score_thr=0.3,
    show=False
)

print("结果已保存")