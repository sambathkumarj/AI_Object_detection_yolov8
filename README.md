# AI_Object_detection_yolov8
DETECTION OF OBJECT USING AI (YOLOV8_ULTRALYTICS)

prerequisite:
Hardware:
Nvidia Jetson orin developer kit(recomputerj4012)
Raspberry-pi camera (CSI or USB camera)
Software:
Roboflow - DataSet
Ultralytics - Model


SETUP FOR NVIDIA JETSON ORIN DEVELOPER KIT

1.2. NVIDIA SDK Manager

https://developer.nvidia.com/sdk-manager
    • NVIDIA Jetson Orin Nano Developer Kit
A Linux host computer running Ubuntu Linux x64 version 20.04 or 18.04 is required to run SDK Manager. Detailed instructions can be found here: 
1.3.1. Install JetPack Components on Jetson Linux

This step assumes your Jetson developer kit has been flashed with and is running L4T 35.3.1. The following commands will install all other JetPack components that correspond to your version of Jetson Linux L4T: 
sudo apt update
sudo apt install Nvidia-jetpack
To view individual Debian packages that are part of the Nvidia-jetpack meta-package, enter the command
Refer to the NVIDIA Jetson Linux Developer Guide for details about L4T-specific Debian packages. 
If disk space is limited, use these commands: 
sudo apt update
apt depends nvidia-jetpack | awk '{print $2}' | xargs -I {} sudo apt install -y {}
1.3.2. Upgrade JetPack
To upgrade from previous JetPack 5.x releases, first edit etc/apt/sources.list.d/nvidia-l4t-apt-source.list to point to the 35.4 repo (just change the version to r35.4 in both lines). Next, use the following commands, then physically reboot the system: 
sudo apt update
sudo apt dist-upgrade
sudo apt install --fix-broken -o Dpkg::Options::="--force-overwrite"


Install DeepStream :
If you install DeepStream using SDK manager, you need to execute the below commands which are additional dependencies for DeepStream, after the system boots up
sudo apt install \
libssl1.1 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev
Install Necessary Packages :

Step 1. Access the terminal of Jetson device, install pip and upgrade it

      sudo apt update
      sudo apt install -y python3-pip
      pip3 install --upgrade pip

Step 2. Clone the following repo
git clone https://github.com/ultralytics/ultralytics.git

Step 3. Open requirements.txt
cd ultralytics
vi requirements.txt

Step 4. Edit the following lines. Here you need to press i first to enter editing mode. Press ESC, then type :wq to save and quit

# torch>=1.7.0
# torchvision>=0.8.1

Step 5. Install the necessary packages

pip3 install -r requirements.txt

If the installer complains about outdated python-dateutil package, upgrade it by
pip3 install python-dateutil --upgrade

Install PyTorch and Torchvision : 

PyTorch v1.11.0
Supported by JetPack 5.0 (L4T R34.1.0) / JetPack 5.0.1 (L4T R34.1.1) / JetPack 5.0.2 (L4T R35.1.0) with Python 3.8
file_name: torch-1.11.0-cp38-cp38-linux_aarch64.whl URL: https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl

PyTorch v1.12.0
Supported by JetPack 5.0 (L4T R34.1.0) / JetPack 5.0.1 (L4T R34.1.1) / JetPack 5.0.2 (L4T R35.1.0) with Python 3.8
file_name: torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl URL: https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl


Step 1. Install torch according to your JetPack version in the following format

sudo apt-get install -y libopenblas-base libopenmpi-dev
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl -O torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl


Step 2. Install torchvision depending on the version of PyTorch that you have installed. For example, we chose PyTorch v1.12.0, which means, we need to choose Torchvision v0.13.0

sudo apt install -y libjpeg-dev zlib1g-dev
git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install –user

Here is a list of the corresponding torchvision versions that you need to install according to the PyTorch version:
    • PyTorch v1.11 - torchvision v0.12.0
    • PyTorch v1.12 - torchvision v0.13.0

Training model using Roboflow :

https://wiki.seeedstudio.com/YOLOv5-Object-Detection-Jetson/#annotate-dataset-using-roboflow

ROBOFLOW – DATASET

ROBOFLOW :
			  Roboflow is a computer vision platform that allows users to build computer vision models faster and more accurately through the provision of better data collection, preprocessing, and model training techniques.

STEP 1: Create Project – Dataset

STEP 2: Upload or drop (images or Video), set framerate as required (for ex: uploaded video and set 1 frame per second) 

STEP 3: After the video is converted into frame, we want to annotate the object in the frame.

STEP 4: After completing annotation for all the objects in the datasets, Then, there are options like preprocessing and augmentation  addition features,if required you may add to in dataset and finally generate the Dataset. For More details
 
Refer link:- https://docs.roboflow.com/

STEP 5 (a): After generating the dataset, It splits the dataset into train_set, Valid_set, and test_set.

STEP 5 (b): The below images show that the pictures in the datasets have been successfully trained.

STEP 6 (a): You can view and verify the  train_set, Valid_set, and test_set in the dataset, if required you can reset and train the dataset again.

STEP 6 (b): Finally dataset is ready for deployment in Ultralytics to create the model. 



ULTRALYTICS – MODEL

ULTRALYTICS :
				Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility.

				 Ultralytics YOLOv8 provides a unified framework for training models for performing object detection, instance segmentation, and image classification. This means that users can use a single model for all three tasks, simplifying the training process

STEP 1:  Create the link between Roboflow with Ultralytics using 
API key provided in the Roboflow, Copy the API key and paste it into the Ultralytics Setting -> Integrations -> Add API key. It creates a linked workplace in Ultralytics.
STEP 2: After the workspace is linked in Ultralytics, Upload the dataset that we downloaded from Roboflow as a .zip file.
STEP 3: While uploading the datasets, you want to select any one of the model formats. (For Ex: Here, I selected yolov5pytorch)
Then, start the training process by clicking on Train Model.



GOOGLE CO-LAB – MODEL RUNTIME

STEP 4 (a): Copy the code, from the last steps and paste it on the google co-lab and run the code.
STEP 4 (b): Below output shows, that the model is successfully
trained and ready for deployment.

STEP 5: After the code is running, the Model is deployed successfully in Ultralytics 
STEP 6: Test the model using the images.

DeepStream Configuration for YOLOv8 :

Step 1. Clone the following repo
cd ~
git clone https://github.com/marcoslucianops/DeepStream-Yolo
Step 2. Checkout the repo to the following commit
cd DeepStream-Yolo
git checkout 68f762d5bdeae7ac3458529bfe6fed72714336ca
Step 3. Copy gen_wts_yoloV8.py from DeepStream-Yolo/utils into ultralytics directory
cp utils/gen_wts_yoloV8.py ~/ultralytics

Step 4. Inside the ultralytics repo, download pt file from YOLOv8 releases (example for YOLOv8s)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

#Add_your_own_.pt_files_from_the_model

NOTE: You can use your custom model, but it is important to keep the YOLO model reference (yolov8_) in your cfg and weights/wts filenames to generate the engine correctly.
    • Step 5. Generate the cfg, wts and labels.txt (if available) files (example for YOLOv8s)
python3 gen_wts_yoloV8.py -w yolov8s.pt

Step 6. Copy the generated cfg, wts and labels.txt (if generated) files into the DeepStream-Yolo folder

cp yolov8s.cfg ~/DeepStream-Yolo
cp yolov8s.wts ~/DeepStream-Yolo
cp labels.txt ~/DeepStream-Yolo

Step 7. Open the DeepStream-Yolo folder and compile the librarycd ~/DeepStream-Yolo
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo  # for DeepStream 6.2/ 6.1.1 / 6.1
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo  # for DeepStream 6.0.1 / 6.0

Step 8. Edit the config_infer_primary_yoloV8.txt file according to your model (example for YOLOv8s with 80 classes)

[property]
...
custom-network-config=yolov8s.cfg
model-file=yolov8s.wts
...
num-detected-classes=80                                    #depend_upon_classes_creating_in_model
...

Step 9. Edit the deepstream_app_config.txt file
...
[primary-gie]
...
config-file=config_infer_primary_yoloV8.txt



Step 10. Change the video source in deepstream_app_config.txt file. Here a default video file is loaded as you can see below
...
[source0]
...
uri=file://home/nvidia/DeepStream-Yolo/(video_name)basana.mp4

Run the Inference :
deepstream-app -c deepstream_app_config.txt

 
