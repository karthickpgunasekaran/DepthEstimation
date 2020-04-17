# DepthEstimation
This is an unofficial re-implementation of [Digging Into Self-Supervised Monocular Depth Estimation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.pdf). Current approaches for 3D bounding box estimation relyon  3D  data  from  LIDAR  sensors  which  are  expensive  togather and process computationally.  In this paper we explore the possibility of using depth estimation networks to produce depth maps that are fed to a Frustum PointNet network for the final estimation of 3D bounding box parameters. We investigate the employment of an end-to-end architecture such that 3D bounding box estimation can be donefrom RGB data by generating depth maps using self supervised learning. 

## How to run it?
```
python Main.py --epochs 5 --lr 1e-2 --do_train True --do_eval False --batch_size 12  --input_data "path-to-input" --output_data "path-to-output"
arguments: 
--epochs      - No of epochs the training should be taking place
--batch_size  - Size of each training batch
--lr          - Learning Rate
--do_train    - If training needs to be carried out
--do_eval     - If eval needs to be carried out
--input_data  - Path to images and depth maps
--output_dir  - output folder to save the model and results
```

