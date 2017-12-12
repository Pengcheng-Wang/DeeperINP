#!/usr/bin/env bash

### Train action prediction models
#cd ../
### The following experiment compare the effects of data augmentation without batch normalization and dropout
#for s in `seq 1 10`;
#do
#    ## Train models without data augmentation
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ];
#        then
#            g=1
#        fi
#        th userSimMain.lua -trType ac -save seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -rnnHdSizeL1 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 0 -seed $(($s)) &
#    done
#    wait
#    ## Train models with data augmentation
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        th userSimMain.lua -trType ac -save seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 1 -seed $(($s)) &
#    done
#    wait
#done

#### The following experiment compare the effects of data augmentation without batch normalization but with dropout
#for s in `seq 1 2`;
#do
#    ## Train models without data augmentation
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ];
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -rnnHdSizeL1 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 0 -seed $(($s)) &
#        else
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -rnnHdSizeL1 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    ## Train models with data augmentation
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 1 -seed $(($s)) &
#        else
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#done

# ### The following experiment compare the effects of data augmentation without batch normalization but with dropout
# for s in `seq 1 1`;
# do
#     # ## Train models without data augmentation
#     # for t in `seq 1 5`;
#     # do
#     #     g=0
#     #     if [ $t -eq 2 ];
#     #     then
#     #         g=1
#     #     fi
#     #     if [ $t -eq 1 ];
#     #     then
#     #         th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -rnnHdSizeL1 64 -rnnHdSizeL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 0 -seed $(($s)) &
#     #     else
#     #         th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -rnnHdSizeL1 64 -rnnHdSizeL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#     #     fi
#     # done
#     # # wait
#     ## Train models with data augmentation
#     for t in `seq 1 5`;
#     do
#         g=0
#         if [ $t -eq 2 ]
#         then
#             g=1
#         fi
#         if [ $t -eq 1 ];
#         then
#             th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 64 -rnnHdSizeL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#         else
#             th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 64 -rnnHdSizeL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#         fi
#     done
#     wait
# done

## Test for different 2-layer lstm size
#s=1
#for lsz in 16 32 64 128
#do
#    echo 'in round' ${lsz}
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 2 sets'
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 4 sets'
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 6 sets'
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 8 sets'
#done

## Test for different multi-layer lstm models, with data augmentation
#s=1
#for lsz in 16 24 32
#do
#    echo 'in round' ${lsz}
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 2 sets'
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 4 sets'
#done

## Test for different multi-layer lstm models, without data augmentation
#s=1
#for lsz in 16 24 32
#do
#    echo 'in round' ${lsz}
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 2 sets'
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 450  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.07 -uSimLstmBackLen 3 -actPredDataAug 0 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 4 sets'
#done

## Test for different multi-layer lstm models, with data augmentation, and random noise added into data points
#s=1
#for lsz in 16 24 32
#do
#    echo 'in round' ${lsz}
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.04762 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 3L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 3 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.04762 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 2 sets'
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.04762 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save 4L${lsz}_${lsz}_lstm_b3/rnndrop.07/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 $(($lsz)) -rnnHdSizeL2 $(($lsz)) -rnnHdLyCnt 4 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 550  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.04762 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 4 sets'
#done

## Test for different multi-layer lstm models, with data augmentation, and random noise added into data points. Nov 19, 2017
#s=1
#for rnnHdLc in 1 2
#do
#    for recD in 3 5 8
#    do
#        echo 'in round' ${recD}
#        for t in `seq 1 5`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        for t in `seq 1 5`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets'
#        echo 'in round' ${recD}
#        for t in `seq 1 5`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        for t in `seq 1 5`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.3/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.3 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}/rnndrop.3/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1200  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.3 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 4 sets'
#    done
#done

## Test for different multi-layer lstm models, with data augmentation, and random noise added into data points. Nov 21, 2017
#s=1
#for rnnHdLc in 1 2
#do
#    for recD in 3 5 8
#    do
#        echo 'in RHN round' ${recD}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in `seq 1 2`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        for t in `seq 1 2`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        echo 'in round' ${recD}
#        for t in `seq 1 2`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        for t in `seq 1 2`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.3/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.3 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType ac -save rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.3/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 3500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.3 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 4 sets in RHN'  ${recD}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done
#wait
#for rnnHdLc in 1 2 3 4
#do
#    echo 'in Bayesian LSTM round' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.3/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.3 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save blstm_hdlc_${rnnHdLc}/rnndrop.3/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.3 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 4 sets in Bayesian LSTM' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#done

#### This is a simple test for GridLSTM. I'm not going to use GPU at this time, because it seems utilizing GPU slows down calculation right now. Not sure if we can improve it by moving entire dataset onto GPU early
#s=1
#for rnnHdLc in 1 2 3 4
#do
#    echo 'in Bayesian GridLSTM round(layer)' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0 #1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 4000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.0/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 4000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0 #1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 4000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 4000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0 #1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save girdlstm_hdlc_${rnnHdLc}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 4000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 4000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 3 sets in Bayesian GridLSTM' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#done


#### Dec 4, 2017, Rerun the prior experiment
#s=1
#for rnnHdLc in 3 4
#do
#    echo 'in Bayesian GridLSTM round(layer)' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0 #1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    for t in `seq 1 2`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0 #1
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        else
#            th userSimMain.lua -trType ac -save gridlstm_hdlc_${rnnHdLc}/rnndrop.2/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.2 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 3 sets in Bayesian GridLSTM' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#done



## Try to do score/outcome prediction using rhn models, without using multi-task structure
#s=1
#for rnnHdLc in 1 2
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in RNN round' ${recD}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in `seq 1 2`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0 #1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save mul_rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 5000  -uSimShLayer 1 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            else
#                th userSimMain.lua -trType sc -save mul_rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 5000  -uSimShLayer 1 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in RHN'  ${recD}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


## Dec 11, 2017.
## Try to do score/outcome prediction using rhn models, without using multi-task structure.
#s=1
#for rnnHdLc in 1 2
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in RNN round' ${recD}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in `seq 1 2`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) &
#            else
#                th userSimMain.lua -trType sc -save sc_rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in RHN'  ${recD}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done

# Dec 11, 2017.
# Try to do score/outcome prediction using cnn moe models
s=1
for moeExpCnt in 16 24 32
do
    for recD in 5
    do
        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
        for t in 1 2 3 4 5
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=0
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType sc -save sc_cnnmoe_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 1 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 3 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
            else
                th userSimMain.lua -trType sc -save sc_cnnmoe_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 1 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 3 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            fi
        done
        wait
        echo 'done with 3 sets in CNN-moe'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done