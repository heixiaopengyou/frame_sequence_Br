The dataset is a subset of the AImageLab Briareo dataset cited below;

@article{briareo2019,
  title={Hand Gestures for the Human-Car Interaction: the {B}riareo dataset},
  author={Manganaro, Fabio and Pini, Stefano and Borghi, Guido and Vezzani, Roberto and Cucchiara, Rita},
  journal={20th International Conference on Image Analysis and Processing (ICIAP)},
  year={2019}
}

The dataset is extracted from the ToF depth data (171x224 frames) stored as npz files.
The data is preprocessed in python before saving it in a .mat format for use in matlab.
The preprocessing includes selecting 30 frames of relevance for each activity (each gesture).
There are 12 gestures in the dataset, from 'g00' to 'g11'.

The .mat files (frames) are to be standardized and resampled in matlab to meet the output size of the ToF sensor proposed by ST Microelectronics, 
before setting up a datastore and feeding the data to the neural network.

Two models shall be proposed, a 2D convolutional neural network from scratch and a pretrained network (alexnet).

After model shall be fine tuned, and after considerable accuracy, prunned and quantized.

The STM32 Cloud shall be used for benchmarking and inference time measured.
Several quantization adjustments shall be made in order to reduce computational cost and facilitate deployability on board.
