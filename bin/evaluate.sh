# Best sigma for only depth: 0.25
# Best sigma for only bilateral: 0.25, 0.5, 1 - no big difference
# Best sigma for edge potential: >=1
for sequence in 5
do
#    for sigma in 0.25 #0.5 1
#    do
        for radius in 0.15 #0.075 0.15 0.3 0.6 1 2
        do 
            sequence_name="sequence0$sequence"
            # Remove segmentation from previous run
            rm ~/cs231a/$sequence_name/*_segmentation.dat
            rm ~/cs231a/$sequence_name/*_bilateral.dat
            rm ~/cs231a/$sequence_name/*_dist.dat
            # Copy seed frame
            cp ~/cs231a/seed/*_segmentation.dat ~/cs231a/$sequence_name
            # Run segmentation
            export DEPG_SIGMA=0.25
            export BILATERAL_SIGMA=0.1
            export NORMAL_SIGMA=0.6
            export EDGE_SIGMA=1
            export DEPG_W=0.2247346990175741 # dist to prev foreground
            export BILATERAL_W=0.2875562529894315
            export DIST_W=0.5929136272571571 # 3d distance edge potential
            export COLOR_W=0.0001327019800154046
            export NORMAL_W=0.5395916472836754
            export RADIUS=$radius
            logfile="logsn0$sequence"
            ./sn_segmenter ~/cs231a/$sequence_name > $logfile
            filename="resultsn0$sequence"
            golden_name="golden0$sequence"
            ./evaluator ~/cs231a/$sequence_name ~/cs231a/$golden_name > $filename
            tail -n 1 $filename
        done
#    done
done
