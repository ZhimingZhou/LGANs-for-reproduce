This repo is for reproducing our results in [“Lipschitz Generative Adversarial Nets”](https://arxiv.org/abs/1902.05687).

We use tensorflow 1.5 with python 3.5.

You can refer to setting_cuda9_cudnn7_tensorflow1.5.sh to build up your environment.

sobolev_cifar10.py:
    
    This code can be used to reproduce results in Table 2, Figure4 and Figure 6.
    
    Run the following code with varying fWeightLip and varying sGAN_type, e.g.,:

        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=0.01 -sGAN_type=exp
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=0.1 -sGAN_type=exp
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=1.0 -sGAN_type=exp
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=10.0 -sGAN_type=exp
    
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=0.1 -sGAN_type=x
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=0.1 -sGAN_type=log_sigmoid
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=0.1 -sGAN_type=sqrt
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=0.1 -sGAN_type=lsgan
        python3 sobolev_cifar10.py -sDataSet=cifar10 -fWeightLip=0.1 -sGAN_type=hinge
    
    By default we use the MaxGP, to switch to GP:

        python3 sobolev_cifar10.py -sDataSet=cifar10 -sGAN_type=exp -fWeightLip=10.0 -bMaxGP=False -sGP_type=gp

    To try the buffered MaxGP, add the following flag: 
    
        -fBufferBatch=0.25
        or
        -fBufferBatch=-0.25
 
        The buffer size equals to fBufferBatch * iBatchSize; 
        The sign of fBufferBatch indicates the way we use the buffer: 
            postive -> extend the batch: batch size for maxgp becomes iBatchSize * (1+fBufferBatch) 
            negative -> insert into the batch: keep the batchsize of gp unchanged.

gan_synthetic4.py:

    This code can be used to reproduce results in Figure 1, Figure 2 and Appendix B.6.
    
    To reproduce Figure 1, run the following code: 
    
        python3 gan_synthetic4.py -iBaseNumFilterD=128 -iBlockPerLayerD=64 -oActD=selu -oOptD=sgd -fLrIniD=1e-4 -sGAN_Type=lsgan -bLip=False -sResultTag=case1
        python3 gan_synthetic4.py -iBaseNumFilterD=128 -iBlockPerLayerD=64 -oActD=selu -oOptD=sgd -fLrIniD=1e-4 -sGAN_Type=lsgan -bLip=False -sResultTag=case2
        python3 gan_synthetic4.py -iBaseNumFilterD=128 -iBlockPerLayerD=64 -oActD=selu -oOptD=sgd -fLrIniD=1e-4 -sGAN_Type=lsgan -bLip=False -sResultTag=case3       
    
    To reproduce Figure 2, run the following code:

        python3 gan_synthetic4.py -iBaseNumFilterD=1024 -iBlockPerLayerD=4 -oActD=relu -oOptD=adam -fLrIniD=1e-4 -bLip=True -sGAN_Type=x -sResultTag=case3
        python3 gan_synthetic4.py -iBaseNumFilterD=1024 -iBlockPerLayerD=4 -oActD=relu -oOptD=adam -fLrIniD=1e-4 -bLip=True -sGAN_Type=log_sigmoid -sResultTag=case3
        python3 gan_synthetic4.py -iBaseNumFilterD=1024 -iBlockPerLayerD=4 -oActD=relu -oOptD=adam -fLrIniD=1e-4 -bLip=True -sGAN_Type=sqrt -sResultTag=case3
        python3 gan_synthetic4.py -iBaseNumFilterD=1024 -iBlockPerLayerD=4 -oActD=relu -oOptD=adam -fLrIniD=1e-4 -bLip=True -sGAN_Type=exp -sResultTag=case3

gan_synthetic4.3.py

    This code can be used to reproduce Figure 3. 
    
        python3 gan_synthetic4.3.py -n=2  # using two cifar10 images as P_r, which can get a quick result.
        python3 gan_synthetic4.3.py -n=10 # using ten cifar10 images as P_r, which is the same setting as Figure 3.    
   
We use training_curve_draw.py to draw the training curves and select random samples.
