import numpy as np
import cv2


# down sample an image by an integer factor
# link: https://api.video/what-is/chroma-subsampling
def subsample(img, format):
    Y, Cb, Cr = img[:,:,0], img[:,:,1], img[:,:,2]         
    img_h , img_w = Y.shape[0], Y.shape[1]
    
    if format==422:
        # reduce width by 2
        Y_even = Y[:, 0::2].ravel().reshape(Y.size//2, 1) # covert to a column vector
        Y_odd  = Y[:, 1::2].ravel().reshape(Y.size//2, 1)
        subsam_Cb = Cb[:, 0::2].ravel().reshape(Cb.size//2, 1) 
        subsam_Cr = Cr[:, 0::2].ravel().reshape(Cr.size//2, 1) 
        subsampled_img = np.round(np.concatenate([Y_even, subsam_Cb, Y_odd, subsam_Cr], axis=1)).astype("uint8")

    elif format==420:
        # reduce height and width by 2
        Y_0 = Y[0::2, 0::2].ravel().reshape(Y.size//4, 1) # covert to a column vector
        Y_1 = Y[0::2, 1::2].ravel().reshape(Y.size//4, 1)
        Y_2 = Y[1::2,0::2].ravel().reshape(Y.size//4, 1) 
        Y_3 = Y[1::2, 1::2].ravel().reshape(Y.size//4, 1)
        subsam_Cb  = Cb[0::2, 0::2].ravel().reshape(Cb.size//4, 1) 
        subsam_Cr  = Cr[0::2, 0::2].ravel().reshape(Cr.size//4, 1) 
        subsampled_img = np.round(np.concatenate([Y_0,Y_1, subsam_Cb, Y_2,Y_3, subsam_Cr], axis=1)).astype("uint8")

    else:
        # format = 444
        Y  = Y.ravel().reshape(Y.size,1)
        Cb = Cb.ravel().reshape(Cb.size,1) 
        Cr = Cr.ravel().reshape(Cr.size,1)    
        subsampled_img = np.round(np.concatenate([Y, Cb, Cr], axis=1)).astype("uint8")
    
    print(subsampled_img.shape)
    raw_wb = open("./YCbCr_{}_{}x{}_raw.raw".format(format, img_w, img_h), 'wb')
    subsampled_img.flatten().tofile(raw_wb)
    raw_wb.close()
    return subsampled_img 

def get_ycbcr(raw_img, shape):

    if raw_img.shape[1]==4:
        # format 422
        Y  = np.zeros(shape)
        Y_even = np.stack(np.hsplit(raw_img[:,0], shape[0]), axis=0)
        Y_odd = np.stack(np.hsplit(raw_img[:,2], shape[0]), axis=0)
        Y[:,0::2] = Y_even
        Y[:,1::2] = Y_odd

        Cb  = np.zeros(shape)
        subsam_Cb = np.stack(np.hsplit(raw_img[:,1], shape[0]), axis=0)
        Cb[:,0::2], Cb[:,1::2] = subsam_Cb, subsam_Cb
    
        Cr  = np.zeros(shape)
        subsam_Cr = np.stack(np.hsplit(raw_img[:,3], shape[0]), axis=0)
        Cr[:,0::2],Cr[:,1::2] = subsam_Cr, subsam_Cr

    elif raw_img.shape[1]==6:
        # format 420
        Y  = np.zeros(shape)
        Y[0::2, 0::2] = np.stack(np.hsplit(raw_img[:,0], shape[0]//2), axis=0)
        Y[0::2, 1::2] = np.stack(np.hsplit(raw_img[:,1], shape[0]//2), axis=0)
        Y[1::2, 0::2] = np.stack(np.hsplit(raw_img[:,3], shape[0]//2), axis=0)
        Y[1::2, 1::2] = np.stack(np.hsplit(raw_img[:,4], shape[0]//2), axis=0)

        Cb  = np.zeros(shape)
        subsam_Cb = np.stack(np.hsplit(raw_img[:,2], shape[0]//2), axis=0) 
        Cb[0::2, 0::2], Cb[0::2, 1::2] = subsam_Cb, subsam_Cb
        Cb[1::2, 0::2], Cb[1::2, 1::2] = subsam_Cb, subsam_Cb       

        Cr  = np.zeros(shape)
        subsam_Cr = np.stack(np.hsplit(raw_img[:,5], shape[0]//2), axis=0) 
        
        Cr[0::2, 0::2], Cr[0::2, 1::2] = subsam_Cr, subsam_Cr
        Cr[1::2, 0::2], Cr[1::2, 1::2] = subsam_Cr, subsam_Cr
        
    else:    
        # format 444
        Y  = raw_img[:,0].reshape(shape)
        Cb = raw_img[:,1].reshape(shape)
        Cr = raw_img[:,2].reshape(shape)
    
    return np.stack([Y, Cb, Cr], axis= 2)

def RGB_to_YCbCr(path, format=444):
    img = cv2.imread(path)
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # convert img
    Y  = 0.2990*R + 0.5870*G + 0.1140*B
    Cb = 127.5 - 0.1687*R - 0.3313*G + 0.5000*B
    Cr = 127.5 + 0.5000*R - 0.4187*G -0.0813*B
    
    # ycbcr_img = np.round(np.stack([Y, Cr, Cb], axis=2)).astype("uint8")         
    ycbcr_img = np.stack([Y, Cb, Cr], axis=2)         

    # subsample
    ycbcr = subsample(ycbcr_img, format)
    return ycbcr

def YCbCr_to_RGB(path):
    """Note: Constants are slightly different(cv2 uses 1.403 instead of 1.4020 in R) from the ones used in cv2"""
    format = int(path.split("/")[-1].split("_")[1])
    img_w  = int(path.split("/")[-1].split("_")[2].split("x")[0])
    img_h  = int(path.split("/")[-1].split("_")[2].split("x")[1])

    if format == 422:
        raw_img = np.fromfile(path, dtype=np.uint8).reshape(img_h*img_w//2, 4)
    elif format == 420:    
        raw_img = np.fromfile(path, dtype=np.uint8).reshape(img_h*img_w//4, 6)
    else:
        raw_img = np.fromfile(path, dtype=np.uint8).reshape(img_h*img_w, 3)
        print(raw_img.shape)
    
    print(raw_img.shape)
    img = get_ycbcr(raw_img, (200, 150))
    print(img.shape)
    
    Y, Cb, Cr = img[:,:,0], img[:,:,1], img[:,:,2]
    R = Y + 1.4020*(Cr-127.5)
    G = Y - 0.3441*(Cb-127.5) - 0.7141*(Cr-127.5)
    B = Y + 1.7720*(Cb-127.5)
    bgr_img = np.round(np.clip(np.stack([B, G, R], axis=2),0,255)).astype("uint8")
    cv2.imwrite("./yuv2rgb_{}.png".format(format), bgr_img) 
    return 


###################################################

# --------------------------------------    
path = "./bird_150x200.jpeg"
ycbcr = RGB_to_YCbCr(path, 420)
# --------------------------------------
yuv_path = "YCbCr_420_150x200_raw.raw"
YCbCr_to_RGB(yuv_path)
#---------------------------------------