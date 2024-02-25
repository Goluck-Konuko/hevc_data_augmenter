# On-the-Fly Data Augmentation for Training Neural Networks to Perform HEVC Intra Prediction

This module uses the standard HEVC implementation for Intra-DC, Planar, and 33 angular directions.

## Data Preparation

The `Augmenter` class allows for flexible data input methods:

- `filepath`: Specify the file path when instantiating the augmenter object.
- `odp`: Supply an ODP (Original Decoded Picture) as a NumPy array.
- `block_size`: Specify the block size.
- `bit_depth`: Specify the bit depth.

Example:

```python
from hevc_augmenter import Augmenter 
augmenter = Augmenter(odp=[optional], filepath=[optional], diskpath=[optional], block_size=[optional], bit_depth=[optional]) 
```

## Usage
a) Check the Acquired Reference Samples
```
left, top, context, original_pu = augmenter.read_context()
print('Left reference samples:', left)
print('Top reference samples:', top)
```

b) Check the Result of Reference Sample Interpolation
```
left, top = augmenter.interpolation()
print('Left reference samples:', left)
print('Top reference samples:', top)
```

c) Check Reference Samples After Linear Filtering
```
left, top = augmenter.filter_reference_array()
print('Left reference samples:', left)
print('Top reference samples:', top)

d) DC Prediction
```
prediction = augmenter.intra_prediction_dc()
print(prediction)
```
e) Planar Prediction
prediction = augmenter.intra_prediction_planar()
print(prediction)

f) Angular Prediction
prediction = augmenter.predict_one(mode=[optional])
print(prediction)

g) Run all prediction modes
prediction = augmenter.predict_all()
print(prediction)
