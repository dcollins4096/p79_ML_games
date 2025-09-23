


from importlib import reload
import go
reload(go)
import pdb  
import plotter
reload(plotter)

#go.ploot(err_Grad=1, err_SSIM=1,err_Pear=1)#,err_Power=1, err_Multi=1)
#go.go(err_Grad=1, err_SSIM=1,err_Pear=1)#,err_Power=1, err_Multi=1)
#go.go(err_Grad=1, err_SSIM=1,err_Pear=1, err_Power=1, err_Multi=1, err_Grad=1)#,err_Power=1, err_Multi=1)

#go.go(err_Grad=1, err_SSIM=1,err_Pear=1, err_Power=1, err_Multi=1,pool_type='avg')
#go.go(err_Grad=1, err_SSIM=1,err_Pear=1, err_Power=1, err_Multi=1,pool_type='max')
#go.go(err_Grad=1, err_SSIM=1,err_Pear=1, err_Power=1, err_Multi=1,pool_type='avgmax')
#go.go(err_Grad=1, err_SSIM=1,err_Pear=1, err_Power=1, err_Multi=1,pool_type='learned')
if 0:
    go.go(err_Power=1)
    go.go(err_Pear=1)
    go.go(err_Multi=1)
    go.go(err_Grad=1)
    go.go(err_SSIM=1)
go.go(err_Grad=1, err_SSIM=1,err_Pear=1, err_Power=1, err_Multi=1,pool_type='max', rotation_prob=0.1)
