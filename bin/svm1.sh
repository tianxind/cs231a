


for skip in 0
do 
 	for sequence in 1 2 3 4 5 6
	do
	    sequence_name="sequence0$sequence"
	    # Remove segmentation from previous run
	    rm ~/cs231a/$sequence_name/*_segmentation.dat
	    rm ~/cs231a/$sequence_name/*_bilateral.dat
	    rm ~/cs231a/$sequence_name/*_dist.dat
	    # Copy seed frame
	    cp ~/cs231a/seed/*_segmentation.dat ~/cs231a/$sequence_name
	done


 
    # Run segmentation
    export DEPG_SIGMA=0.5
    export BILATERAL_SIGMA=0.1
    export NORMAL_SIGMA=0.1
    export EDGE_SIGMA=1
    export DEPG_W=1 # dist to prev foreground
    export BILATERAL_W=1
    export DIST_W=1 # 3d distance edge potential
    export COLOR_W=1
    export NORMAL_W=1
    export RADIUS=0.15
    logfile="svm_leaveout_0$skip"
    ./loop ~/cs231a/sequence 1 1 $skip > $logfile
    tail -n 10 $logfile
done
 

