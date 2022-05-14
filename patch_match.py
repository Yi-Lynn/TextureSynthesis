import numpy as np
import matplotlib.pyplot as plt
import cv2

def computeSSD(ia,ja,ib,jb):
    
    G = cv2.getGaussianKernel(ksize=patchSize,sigma=6)
    G = G * G.T
    
    Apatch = A_pad[ia:ia+patchSize,ja:ja+patchSize,:]
    ApatchMask = A_mask[ia:ia+patchSize,ja:ja+patchSize]
    Bpatch = B_pad[ib:ib+patchSize,jb:jb+patchSize,:]
    
    squared_diff = np.square(Apatch-Bpatch)
    squared_diff = np.mean(squared_diff,axis=2)
    SSD = np.sum(np.multiply(G*ApatchMask,squared_diff))
    
    return SSD


def computeNewSSDandOFF(i,j,offset1):
    B_i = int(i+offset1[1])
    B_j = int(j+offset1[0])
    
    # if (x,y)+f(x,y) exceeds the boundary of B, set it to the boundary pixel
    if B_i>Bsize[0]-1:     
        B_i = Bsize[0]-1
    elif B_i<0:
        B_i = 0
    if B_j>Bsize[1]-1:     
        B_j = Bsize[1]-1
    elif B_j<0:
        B_j = 0
        
    new_offset = [B_j-j,B_i-i]  
    new_ssd = computeSSD(i,j,B_i,B_j)
    
    return new_ssd,new_offset


# Zero pad the images and the image masks
def img_padding(A, B):
    A_pad = np.zeros([Asize[0]+2*patchL,Asize[1]+2*patchL,3])
    A_pad[patchL:-patchL,patchL:-patchL,:] = A
    B_pad = np.zeros([Bsize[0]+2*patchL,Bsize[1]+2*patchL,3])
    B_pad[patchL:-patchL,patchL:-patchL,:] = B

    A_mask = np.zeros([Asize[0]+2*patchL,Asize[1]+2*patchL])
    A_mask[patchL:-patchL,patchL:-patchL] = np.ones([Asize[0],Asize[1]]) 
    B_mask = np.zeros([Bsize[0]+2*patchL,Bsize[1]+2*patchL])
    B_mask[patchL:-patchL,patchL:-patchL] = np.ones([Bsize[0],Bsize[1]])

    return A_pad, B_pad, A_mask, B_mask


# Initialization of the nearest-neighbor field (NNF): for each patch(i,j), NNF(i,j)=(u,v,D)
def initialization(): 
    NNF = np.zeros([Asize[0],Asize[1],3])
    for row in range(Asize[0]):
        for col in range(Asize[1]):
            
            # locate a randow position in B
            rand_row = np.random.randint(0,Bsize[0])
            rand_col = np.random.randint(0,Bsize[1])
            
            # calculate offset and ssd
            u = rand_col-col
            v = rand_row-row
            ssd = computeSSD(row,col,rand_row,rand_col)
            NNF[row,col] = u,v,ssd 
    return NNF
    

# Propagation 
def propagation():

    for iter in range(1,ITER_MAX+1):

        is_odd = 1 if iter%2==1 else 0
        print("iter:",iter)

        for i in range(Asize[0]):
            for j in range(Asize[1]):
        
                #### propagate from top and left
                if is_odd:

                    # offset for the center, top, left patch
                    neighbors_offset = [NNF[i,j,0:2], NNF[max(0,i-1),j,0:2], NNF[i,max(0,j-1),0:2]]
                        
                    # find new_offset and new B patch 
                    offset1 =  neighbors_offset[1] 
                    ssd1,offset1 = computeNewSSDandOFF(i,j,offset1)
                    offset2 =  neighbors_offset[2] 
                    ssd2,offset2 = computeNewSSDandOFF(i,j,offset2)
                    
                    # determine the pixel with the smallest ssd
                    neighbors_offset = [NNF[i,j,0:2],offset1,offset2]
                    neighbors_ssd = [NNF[i,j,2],ssd1,ssd2]  
                    idx = neighbors_ssd.index(min(neighbors_ssd))        
                    
                    if idx!=0:
                        NNF[i,j,0:2] =  neighbors_offset[idx] # update offset 
                        NNF[i,j,2] = neighbors_ssd[idx]       # update ssd
                        
                #### propagate from down and right
                if not(is_odd):
                    
                    i = Asize[0]-i-1
                    j = Asize[1]-j-1
    
                    # offset for the center, down, right patch
                    neighbors_offset = [NNF[i,j,0:2], NNF[min(Asize[0]-1,i+1),j,0:2], NNF[i,min(Asize[1]-1,j+1),0:2]]
                    
                    # find new_offset and new B patch 
                    offset1 =  neighbors_offset[1] 
                    ssd1,offset1 = computeNewSSDandOFF(i,j,offset1)
                    offset2 =  neighbors_offset[2] 
                    ssd2,offset2 = computeNewSSDandOFF(i,j,offset2)
                    
                    neighbors_offset = [NNF[i,j,0:2],offset1,offset2]
                    neighbors_ssd = [NNF[i,j,2],ssd1,ssd2]  
                    idx = neighbors_ssd.index(min(neighbors_ssd))
                    
                    if idx!=0:
                        NNF[i,j,0:2] =  neighbors_offset[idx] # update offset 
                        NNF[i,j,2] = neighbors_ssd[idx]       # update ssd    


# Visualize the x and y coordinates of the patch correspondence field
def visualization():
    A_constructed = np.zeros(A.shape)
    for i in range(Asize[0]):
        for j in range(Asize[1]):
            B_i = int(i+NNF[i,j,1])
            B_j = int(j+NNF[i,j,0])      
            A_constructed[i,j,:] = B[B_i,B_j,:]
    plt.imshow(A_constructed/255)
    plt.title('Window size = %i'%patchSize)
    plt.show()
    

if __name__ == '__main__':

    patchSize = 9
    patchL = round((patchSize-1)/2)
    ITER_MAX = 4

    A = plt.imread('pic2-1.jpg')
    B = plt.imread('pic2-2.jpg')

    Asize = (A.shape[0],A.shape[1])
    Bsize = (B.shape[0],B.shape[1])

    A_pad, B_pad, A_mask, B_mask = img_padding(A, B)

    NNF = initialization()
    propagation()
    visualization()