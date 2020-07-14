#source activate dlp_new

#cd ~/dlp/PycharmProjects/ROP/RPC

python RPC_server_single_class.py 0 5000 &  # image quality
python RPC_server_single_class.py 1 5001 &  # left right eye
python RPC_server_single_class.py 2 5002 &  # stage
python RPC_server_single_class.py 3 5003 &  # hemorhrage
python RPC_server_single_class.py 4 5004 &  # posterior
#python RPC_server_single_class.py 5 5005 &  # plus classification one stage
python RPC_server_single_class.py 6 5006 &  # plus classification two stages

python RPC_server_blood_vessel_seg.py 10 5010 &  # patch based blood vessel segmentation
python RPC_server_optic_disc_seg.py 11 5011 &  # Mask RCNN optic disc segmentation


python RPC_server_deep_shap.py 0 5100 &  # class type,gpu_no  port no


#sudo lsof -i:5000
