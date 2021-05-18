# Generating the UFF model for TensorRT

It is suggested to create a conda environment first.

1. Install the dependencies

    ``` sh
    # Mask R-CNN dependencies
    pip install -r requirements.txt
    # The UFF converter
    pip install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com uff
    pip install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com graphsurgeon
    ```

1. From the repository root apply the patch to change the channel order in the
   network

    ``` sh
    git apply uff-converter/0001-Update-the-Mask_RCNN-model-from-NHWC-to-NCHW.patch
    ```

    Keep in mind that `demo.py` will not work with the patch.

1. Apply the patch to change the channel order in the UFF converter

    ``` sh
    # For a conda installation (recommended)
    patch ~/.conda/envs/maskrcnn/lib/python3.7/site-packages/uff/converters/tensorflow/converter_functions.py uff-converter/uff_converter.patch
    # For a system installation
    patch /usr/lib/python3.6/dist-packages/uff/converters/tensorflow/converter_functions.py uff-converter/uff_converter.patch
    ```

1. Build the Matterport version of Mask R-CNN normally as shown in the
   [README](../README.md)

    ``` sh
    python setup.py install
    ```

1. Download the pre-trained Keras model

    ``` sh
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
    ```

    The md5sum of the model file is e98aaff6f99e307b5e2a8a3ff741a518.

1. Convert the h5 model to a UFF model

    ``` sh
    cd uff-converter
    python mrcnn_to_trt_single.py -w ../mask_rcnn_coco.h5 -o mrcnn_nchw.uff -p ./config.py
    ```

