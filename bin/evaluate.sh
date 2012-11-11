for sequence in 1 5 6
do
    for sigma in 0.0312 0.0625 0.125
    do 
        sequence_name="sequence0$sequence"
        # Remove segmentation from previous run
        rm ~/cs231a/$sequence_name/*_segmentation.dat
        # Copy seed frame
        cp ~/cs231a/seed/*_segmentation.dat ~/cs231a/$sequence_name
        # Run segmentation
        export DEPG_SIGMA=$sigma
        ./graphcuts_segmenter ~/cs231a/$sequence_name
        filename="result0$sequence$sigma"
        golden_name="golden0$sequence"
        ./evaluator ~/cs231a/$sequence_name ~/cs231a/$golden_name > $filename
    done
done
for sequence in 1 5 6
do
    for sigma in 0.0312 0.0625 0.125
    do 
        filename="result0$sequence$sigma"
        tail -n 1 $filename
    done
done

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
