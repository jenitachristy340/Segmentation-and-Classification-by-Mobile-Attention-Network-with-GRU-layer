import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

no_of_Dataset = 2
def Image_Results():
    Dataset = ['HAM10000', 'PH2Dataset']
    I = [77, 74, 131, 135, 141]
    for n in range(no_of_Dataset):
        Images = np.load('Images_'+str(n+1)+'.npy', allow_pickle=True)
        GT = np.load('GT_'+str(n+1)+'.npy', allow_pickle=True)
        UNET = np.load('Method3_Dataset'+str(n+1)+'.npy', allow_pickle=True)
        RESUNET = np.load('Method4_Dataset'+str(n+1)+'.npy', allow_pickle=True)
        TRANSUNET = np.load('Method2_Dataset'+str(n+1)+'.npy', allow_pickle=True)
        ASUnet = np.load('Method1_Dataset'+str(n+1)+'.npy', allow_pickle=True)
        PROPOSED = np.load('Method5_Dataset'+str(n+1)+'.npy', allow_pickle=True)
        for i in range(len(I)):
            # print(i)
            plt.suptitle("Image Results from  "+ Dataset[n])
            plt.subplot(2, 4, 1)
            plt.title('Original')
            plt.imshow(Images[I[i]])
            plt.subplot(2, 4, 2)
            plt.title('GroundTruth')
            plt.imshow(GT[I[i]])
            plt.subplot(2, 4, 3)
            plt.title('UNET')
            plt.imshow(UNET[I[i]])
            plt.subplot(2, 4, 4)
            plt.title('RESUNET')
            plt.imshow(RESUNET[I[i]])
            plt.subplot(2, 4, 5)
            plt.title('TRANSUNET')
            plt.imshow(TRANSUNET[I[i]])
            plt.subplot(2, 4, 6)
            plt.title('ASUnet++')
            plt.imshow(ASUnet[I[i]])
            plt.subplot(2, 4, 7)
            plt.title('PROPOSED')
            plt.imshow(PROPOSED[I[i]])
            plt.tight_layout()
            # path = "./Results/Image_Results/Image_%s_%s.png" % (n + 1, i + 1)
            # plt.savefig(path)
            plt.show()
            cv.imwrite('./Results/Image_Results/Dataset-'+str(n+1) +'orig-' + str(i + 1) + '.png', Images[I[i]])
            #
            # cv.imwrite('./Results/Image_Results/Dataset-'+str(n+1) +'gt-' + str(i + 1) + '.png', GT[I[i]])
            # cv.imwrite('./Results/Image_Results/Dataset-'+str(n+1) +'unet-' + str(i + 1) + '.png', UNET[I[i]])
            # cv.imwrite('./Results/Image_Results/Dataset-'+str(n+1) +'resunet-' + str(i + 1) + '.png',
            #            RESUNET[I[i]])
            # cv.imwrite('./Results/Image_Results/Dataset-'+str(n+1) +'transunet-' + str(i + 1) + '.png',
            #            TRANSUNET[I[i]])
            # cv.imwrite('./Results/Image_Results/Dataset-'+str(n+1) +'amunetplusplus-' + str(i + 1) + '.png',
            #            ASUnet[I[i]])
            # cv.imwrite('./Results/Image_Results/Dataset-'+str(n+1) +'proposed-' + str(i + 1) + '.png',
            #            PROPOSED[I[i]])


def Sample_Images():
    Dataset = ['HAM10000', 'PH2Dataset']
    for n in range(no_of_Dataset):
        Orig = np.load('Images_'+str(n+1)+'.npy', allow_pickle=True)
        ind = [10, 20, 30, 40, 50, 60]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from  "+ Dataset[n])
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[5]])
        plt.show()
        # cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(0 + 1) + '.png', Orig[ind[0]])
        # cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(1 + 1) + '.png', Orig[ind[1]])
        # cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(2 + 1) + '.png', Orig[ind[2]])
        # cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(3 + 1) + '.png', Orig[ind[3]])
        # cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(4 + 1) + '.png', Orig[ind[4]])
        # cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(5 + 1) + '.png', Orig[ind[5]])


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
