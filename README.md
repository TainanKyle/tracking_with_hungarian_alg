# Tracking Using Detection and Hungarian Algorithm

### Description
A Human Tracking Model which tracks each person with a specific id. The model uses YOLOX to detect people in each frame, and match the id with bounding boxes by using Hungarian Algorithm. It has a great performance in scenes where crowds frequently intersect.

### How to Run
1. Download the pretrained weights of YOLOX and ReID.
2. Modify the path of input, output, and weights.
3. Run `python model.py`.

### Demo Result

https://github.com/TainanKyle/tracking_with_hungarian_alg/assets/150419874/c98a119a-a4dd-4024-9a50-5659ae312b93

