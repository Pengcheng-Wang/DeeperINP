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
#        th userSimMain.lua -trType ac -save seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 0 -seed $(($s)) &
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
#        th userSimMain.lua -trType ac -save seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 1 -seed $(($s)) &
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
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 0 -seed $(($s)) &
#        else
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 0 -seed $(($s)) > /dev/null &
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
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 1 -seed $(($s)) &
#        else
#            th userSimMain.lua -trType ac -save rnndrop.1/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -actPredDataAug 1 -seed $(($s)) > /dev/null &
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
#     #         th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -lstmHdL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 0 -seed $(($s)) &
#     #     else
#     #         th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -lstmHdL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 0 -seed $(($s)) > /dev/null &
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
#             th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -lstmHdL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#         else
#             th userSimMain.lua -trType ac -save 2L64_48_lstm/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -lstmHdL2 48 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -actPredDataAug 1 -seed $(($s)) > /dev/null &
#         fi
#     done
#     wait
# done

s=1
for lsz in 16 32 64 128
do
    echo 'in round' ${lsz}
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    wait
    echo 'done with 2 sets'
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b6/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 6 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    wait
    echo 'done with 4 sets'
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.0/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.15/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    wait
    echo 'done with 6 sets'
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.25/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        else
            th userSimMain.lua -trType ac -save 2L${lsz}_${lsz}_lstm_b3/rnndrop.4/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd $(($lsz)) -lstmHdL2 $(($lsz)) -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 300  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
        fi
    done
    wait
    echo 'done with 8 sets'
done
