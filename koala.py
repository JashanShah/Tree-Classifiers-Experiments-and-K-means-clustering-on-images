import numpy as np
import matplotlib.pyplot as sav
import matplotlib.image as img


input_image = "C:\HW3\hw3_part2_data\Koala.jpg"
kValue = input("Enter the value of K: ")
# Making the string input to int
kValue = int(kValue)
Koala = img.imread(input_image)

counter = 0
pixelValues = Koala.shape[0] * Koala.shape[1]
RGB = np.empty((pixelValues, 3))
for y in range(Koala.shape[0]):
    for j in range(Koala.shape[1]):
        RGB[counter] = Koala[y][j]
        counter += 1

row, col = RGB.shape

RGBUpdated = np.empty((row, col))
total = np.zeros((kValue, col))
distance = np.empty(row)

count = np.zeros(kValue).reshape(kValue, 1)
random = np.random.choice(RGB.shape[0], kValue, replace=None)
centers = RGB[random]


for x in range(10):
    def minimumDistance(X):
        return np.argmin(np.linalg.norm((X - centers), axis=1))
    distance = np.apply_along_axis(minimumDistance, 1, RGB)
    for y in range(row):
        count[int(distance[y])] += 1
        total[int(distance[y])] += RGB[y]
    centers = total / count

for z in range(row):
    RGBUpdated[z] = centers[int(distance[z])]

pic = (np.reshape(RGBUpdated, Koala.shape) / 255)
sav.imsave("newKoala.jpg", pic)
print("New image created.")
