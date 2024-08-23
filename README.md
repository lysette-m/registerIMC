# registerIMC

(In progress)

## Description 

A package for registration/stitching of Imaging Mass Cytometry ROIs using a reference immunofluorescence (IF) image.

## Usage

register_IF_IMC() is the main function for registration. This will output registered/stitched images as .tiff files, generating one image per marker channel. The function assumes that the IMC directory is structured similar to below: 

```bash
.
├── ROI1
│   ├── marker_1.tiff
│   ├── marker_2.tiff
│   ├── marker_3.tiff
│   └── ...
├── ROI2
│   ├── marker_1.tiff
│   ├── marker_2.tiff
│   ├── marker_3.tiff
│   └── ...
└── ...

```

This also requires a pre-processed IF image. There is a default pre-processing function (scaling, rotation and binarisation) within preprocess.py (```preprocess_IF()```), or you can do these steps manually. The registration approach assumes any large discrepancies in orientation between IF and IMC images is first corrected, i.e. 90, 180 degree rotations of the IF image - the images should be in the same 'general' orientation. This level of rotation is included as an option in the default pre-processing function. The scale for the pre-processing function is calculated as IF resolution / IMC resolution.

The markers to register can be extracted from a provided panel metadata file (tab-delimited txt). Alternatively, you can provide a custom list of markers to register (note that the marker labels must be present within the IMC filenames).

The registration process is performed in two steps:

1) Template matching to identify the general location of the IMC image within the IF reference. 
2) Identification of the rotation angle of the IMC image. 

The second step is performed only within the area identified by step 1. This step repeats the template matching, but with small incremental rotations of the IMC image to identify the 'best' rotation angle. 

You can specify the maximum rotation angle to consider with the '```max_angle```' argument, as well as the size of the increment for the rotation search ('```angle_increment```').












