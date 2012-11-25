# Best sigma for only depth: 0.25
# Best sigma for only bilateral: 0.25, 0.5, 1 - no big difference
# Best sigma for edge potential: >=1
for sequence in 4 #2 3 4 5 6
do
    for sigma in 0.25 #0.5 1
    do
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
            export DEPG_W=1
            export COLOR_W=0
            export BILATERAL_W=1
            export EDGE_SIGMA=1
            export RADIUS=$radius
            logfile="logb0$sequence$sigma"
            ./bilateral_segmenter ~/cs231a/$sequence_name > $logfile
            filename="resultb0$sequence$radius$sigma"
            golden_name="golden0$sequence"
            ./evaluator ~/cs231a/$sequence_name ~/cs231a/$golden_name > $filename
            tail -n 1 $filename
        done
    done
done
#for sequence in 4
#do
#    for sigma in 0.25 #0.5 1
#    do
#        for edge_sigma in 0.25 0.5 2
#        do 
#            filename="resultb0$sequence$edge_sigma$sigma"
#            tail -n 1 $filename
#        done
#    done
#done
#for sigma in 0.0625 0.125 0.25
#do 
#    cp ~/cs231a/sequence01/1288572831.002852_segmentation.dat ~/cs231a/sequence01/1288572831.002852_segmentation.dat2
#    rm ~/cs231a/sequence01/*segmentation.dat
#    cp ~/cs231a/sequence01/1288572831.002852_segmentation.dat2 ~/cs231a/sequence01/1288572831.002852_segmentation.dat
#    export DEPG_SIGMA=$sigma
#    ./graphcuts_segmenter ~/cs231a/sequence01
#    filename="result$sigma"
#    ./evaluator ~/cs231a/sequence01 ~/cs231a/golden01 > $filename
#    tail -n 1 $filename
#done
