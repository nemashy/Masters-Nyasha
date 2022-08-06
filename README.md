# Radar-based Multi-Target Classification using Deep Learning

Real-time, radar-based human activity and target recognition has several applications in
various fields. Examples include hand gesture recognition, border and home surveillance,
pedestrian recognition for automotive safety and fall detection for assisted living. This
dissertation sought to improve the speed and accuracy of a previously developed model
classifying human activity and targets using radar data for outdoor surveillance purposes.
An improvement in accuracy and speed of classification helps surveillance systems to provide
reliable results on time. For example, the results can be used to intercept trespassers, poachers
or smugglers.

To achieve these objectives, radar data was collected using a C-band pulse-Doppler radar
and converted to spectrograms using the Short-time Fourier Transform (STFT) algorithm.
Spectrograms of the following classes were utilised in classification: one human walking, two
humans walking, one human running, moving vehicles, a swinging sphere and clutter/noise.
A seven-layer residual network was proposed, which utilised Batch Normalisation (BN),
Global Average Pooling (GAP), and residual connections to achieve a classification accuracy of
92.90% and 87.72% on the validation and test data, respectively. Compared to the previously
proposed model, this represented a 10% improvement in accuracy on the validation data and
a 3% improvement on the test data.

Applying model quantisation provided up to 3.8 times speedup in inference, with a less than
0.4% accuracy drop on both the validation and test data. The quantised model could support
a range of up to 89.91 kilometres in real-time, allowing it to be used in radars that operate
within this range.
