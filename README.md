On the fly data augmentation for training neural networks to perform HEVC intra prediction
This module uses the standard HEVC implementation for Intra-DC, Planar and 33 angular directions.


# 1. DATA PREPARATION
The input method(read_context()) allows the filepath to be specified when instatiating the augmenter object. alternatively, supply an odp as a numpy array or just the decoded block.
example:

from hevc_augmenter import Augmenter 
augmenter = Augmenter(odp=[optional],filepath=[optional],diskpath=[optional],block_size=[optional],bit_depth=[optional]) 

# 2. USAGE
# a)Check the acquired reference samples

left, top,context,original_pu = augmenter.read_context()
print('Left reference samples',left)
print('Top reference samples',top)

# b)Check the result of reference sample interpolation

left,top = augmenter.interpolation()
print('Left reference samples',left)
print('Top reference samples',top)

# c)Check reference samples after linear filtering
left,top = augmenter.filter_reference_array()
print('Left reference samples',left)
print('Top reference samples',top)

# d)Perform DC prediction
prediction = augmenter.intra_prediction_dc()
print(prediction)


# e)Perform Planar prediction
prediction = augmenter.intra_prediction_planar()
print(prediction)

# f)Perform one Angular Prediction
prediction = augmenter.predict_one(mode = [optional])
print(prediction)


# g)Perform all predictions
prediction = augmenter.predict_all()
print(prediction)

## RUN the example.py file to observe the outputs of different augmenter processes