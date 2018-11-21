# Hackathon-Segmentation-with-Sonification-
Image Segmentation with YOLOv3 for identifying objext position and size for creating size dependent pitch sounds.

Direction for using python mpd.py file

-> clone the repository 
-> download the yolo-coco model weights and save the folder as yolo-coco in the same repository directory
-> change current working directory as the repository
-> copy the desired video in the video folder as well as output folder
-> run the following command 
   python3 mod.py -i video/filename.avi -p output/filename.py -y yolo-coco
