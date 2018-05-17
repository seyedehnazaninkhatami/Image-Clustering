# HW 5

# Importing Modules
import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


# Question 1
# Reading Data
def read_scene():
    data_x = misc.imread('Desktop\MachineLearning\HW5\EC\Data\\umass_campus.jpg')
 
    return (data_x)
data_x = read_scene()
print('X = ', data_x.shape)
flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
flattened_image = flattened_image / 255

# Making Grid
fig = plt.figure(figsize=(10,10))
# Number of Clusters
cluster_number=[2,5,10,25,50,75,100,200]
plt.subplot(3,3,1)
name= 'Original Image'
plt.imshow(data_x)
plt.title(name,horizontalalignment='center', y=-0.16)
plt.axis('off')
Reconstruction_Error=np.zeros((8,1))

# Loop for K-means Clustering with 8 different k + Calculating Reconstruction Error
for i in range(1,len(cluster_number) + 1):

    kmeans = KMeans(n_clusters=cluster_number[i-1], random_state=0).fit(flattened_image)
    labels = kmeans.predict(flattened_image)
    centroids = kmeans.cluster_centers_
    labels_mat = np.asmatrix(labels)
    labels_mat = np.transpose(labels_mat)
    flattened_image_clustered = np.zeros((160000,3))
    
    for j in range(0,160000):
        flattened_image_clustered[j,:] = centroids[labels_mat[j]]
    
    # Reconstruction Error     
    Reconstruction_Error[i-1]= np.sqrt((np.power((flattened_image_clustered -  flattened_image),2)).mean()  ) 
    # Reconstructed Image     
    reconstructed_image = flattened_image_clustered.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    # Plotting in the Grid
    plt.subplot(3,3,i+1)
    name= 'K=' + str(cluster_number[i-1])
    plt.imshow(reconstructed_image)
    plt.title(name,horizontalalignment='center', y=-0.16)
    plt.axis('off')
    

plt.savefig("KMeans.jpg")    
plt.show()


# Question 2
# Reading Data
def read_scene():
    data_x = misc.imread('Desktop\\MachineLearning\\HW5\\umass_campus\\umass_campus_100x100x3.jpg')
 
    return (data_x)
data_x = read_scene()
print('X = ', data_x.shape)
flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
flattened_image = flattened_image / 255

# Making Grid
fig = plt.figure(figsize=(10,10))
# Number of Clusters
cluster_number=[2,5,10,25,50,75,100,200]
plt.subplot(3,3,1)
name= 'Original Image'
plt.imshow(data_x)
plt.title(name,horizontalalignment='center', y=-0.16)
plt.axis('off')
Reconstruction_Error=np.zeros((8,1))

# Loop for HAC Clustering with 8 different k + Calculating Reconstruction Error
for i in range(1,len(cluster_number) + 1):

    Hclustering = AgglomerativeClustering(n_clusters=cluster_number[i-1], affinity='euclidean', linkage='ward').fit(flattened_image)
    labe= Hclustering.fit_predict(flattened_image)
    labels_mat = np.asmatrix(labe)
    labels_mat = np.transpose(labels_mat)
    flattened_image_label = np.append(flattened_image,labels_mat,axis=1)
    import pandas 
    Col=['red','green','blue','label']
    df = pandas.DataFrame(flattened_image_label, columns=Col)
    rr = df.groupby('label')['red'].mean()
    gg = df.groupby('label')['green'].mean()
    bb = df.groupby('label')['blue'].mean()
    RGB = pandas.concat([rr,gg,bb],axis=1)
    RGB = RGB.as_matrix()
    flattened_image_clustered = np.zeros((10000,3))
    
    for j in range(0,10000):
        flattened_image_clustered[j,:] = RGB[labels_mat[j]]
        
    # Reconstruction Error
    Reconstruction_Error[i-1]= np.sqrt((np.power((flattened_image_clustered -  flattened_image),2)).mean()  ) 
    # Reconstructed Image    
    reconstructed_image = flattened_image_clustered.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    # Plotting in the Grid
    plt.subplot(3,3,i+1)
    name= 'K=' + str(cluster_number[i-1])
    plt.imshow(reconstructed_image)
    plt.title(name,horizontalalignment='center', y=-0.16)
    plt.axis('off')
    
    
    
    
plt.savefig("HAC.jpg")    
plt.show()