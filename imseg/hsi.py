from numpy import sqrt, arctan2

def rgb_to_hsl(image):
    """Convert rgb image to hsi"""
    hsi = image.copy()
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    alpha = 0.5*(2*R - G - B)
    beta = sqrt(3)/2*(G - B)
    hsi[:,:,0] = arctan2(beta, alpha)
    hsi[:,:,2] = (R + G + B)/3.
    hsi[:,:,1] = 1. - image[:,:,:3].min(axis=2)/hsi[:,:,2]
    return hsi
