import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Parameters
new_h=128  # height of the synthesis image
new_w=128  # weight of the synthesis image
SIGMA = 6  # deviation of the Gaussian Filter
THRES = .1 # threshold when randomly choose a best match

def getNextUnfilledPixel(M,M_pad):
   
    # obtain the boundary map, i.e. the unfilled pixels
    dilated_M = cv2.dilate(M, np.ones((3,3)))
    pixels_map = dilated_M - M  
    pixels_indices = np.nonzero(pixels_map)  # rows/columns of all neighboring unfilled pixels

    N = pixels_indices[0].shape[0]  # total number of neighboring unfilled pixels

    # calculate and store number of neighbors for each unfilled pixels
    num_neighbors = np.zeros(N)
    for i in range(N):
        row = pixels_indices[0][i]
        col = pixels_indices[1][i]
        num_neighbors[i] = np.sum(M_pad[row:row+2*window+1, col:col+2*window+1])

    sorted_order = np.argsort(num_neighbors)[::-1]
    sorted_pixels_indices = ([pixels_indices[0][sorted_order], pixels_indices[1][sorted_order]])
    nextPixelIdx = (sorted_pixels_indices[0][0],sorted_pixels_indices[1][0])

    return nextPixelIdx[0], nextPixelIdx[1]

def computeSSDmatrix(patch, patchMask,  template_pad,  G):
    SSD = np.zeros([h,w])
    for i in range(h):
        for j in range(w):      
            template_patch = template_pad[i:i+2*window+1,j:j+2*window+1,:]
            squared_diff = np.square(template_patch-patch)
            squared_diff = np.mean(squared_diff,axis=2)
            SSD[i,j] = np.sum(np.multiply(G*patchMask,squared_diff))
    return SSD

def randomPick(SSD):
    ssd_min = np.min(SSD)
    err_range = ssd_min*(1.+THRES)
    indices = np.where(SSD <= err_range)
    N = indices[0].shape[0]
    
    # randomly select one best matches
    rand_idx = np.random.choice(np.arange(N),size=1)
    selected_idx = (indices[0][rand_idx], indices[1][rand_idx])
    return selected_idx

def imageFilled(imgMask):
    if (imgMask == 0).any():
        return False  # image synthesis is not completed
    else:
        return True   # image synthesis is completed

def gaussFilter(sigma):
    G = cv2.getGaussianKernel(ksize=2*window+1, sigma=sigma)
    G = G * G.T
    return G

def textureSynthesis(T, window): 

    print("synthesizing...")
        
    # zero-pad the template image and create mask
    T_pad = np.zeros([h+2*window,w+2*window,3])
    T_pad[window:-window,window:-window,:] = T
    T_mask = np.zeros([h+2*window,w+2*window])
    T_mask[window:-window,window:-window] = np.ones([h,w])

    # initalize the image that need to be synthesized and create mask
    E = np.zeros([new_h,new_w,3])
    E[0:h,0:h,:]=T
    M = np.zeros([new_h,new_w])
    M[0:h,0:w]=np.ones([h,w])

    # zero-window on E and M
    E_pad = np.zeros([new_h+2*window,new_w+2*window,3])
    E_pad[window:-window,window:-window,:] = E
    M_pad = np.zeros([new_h+2*window,new_w+2*window])
    M_pad[window:-window,window:-window] = M
    
    while not(imageFilled(M)):
        row,col = getNextUnfilledPixel(M,M_pad)  
        patch = E_pad[row:row+2*window+1, col:col+2*window+1,:]
        patchMask = M_pad[row:row+2*window+1, col:col+2*window+1]
        G = gaussFilter(SIGMA)
        SSD = computeSSDmatrix(patch, patchMask,T_pad,G)
        BestMatch = randomPick(SSD)
        r = BestMatch[0]
        c = BestMatch[1]

        # update image and mask
        E[row,col,:] = T[r,c,:]
        M[row,col] = 1     
        E_pad[row+window, col+window,:] = T[r,c,:]
        M_pad[row+window, col+window] = 1

    return E

if __name__ == '__main__':

    # Load template image
    T = plt.imread('pic1.jpg')
    T = T/255
    h,w,c = T.shape

    # Set window size
    window = 5

    # Texture syntesizing
    start = time.time()
    img = textureSynthesis(T,window)
    end = time.time()

    # Visulize synthesis results and output snthesis time
    plt.figure()
    plt.imshow(img)
    plt.title('window = %i'%(2*window+1))
    plt.show()
    print("Running time: ", end-start)