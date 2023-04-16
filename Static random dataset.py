import matplotlib.pyplot as plt
def main( args) :
    arrayRow = 50
    arrayCol = 2
    randomNumbers = [[0.0] * (arrayCol) for _ in range(arrayRow)]
    x_coordinates = [[0.0] * (arrayCol-1) for _ in range(arrayRow)]
    y_coordinates = [[0.0] * (arrayCol-1) for _ in range(arrayRow)]
    seed = 10
    m = 2**32
    a = 1664525
    c = 1013904223
    xPrev = seed
    i = 0
    while (i < arrayRow) :
        j = 0
        while (j < arrayCol) :
            temp = ((xPrev * a) + c) % m
            randomNumbers[i][j] = int(((temp / m) * 10))
            xPrev = temp
            #print(str(randomNumbers[i][j]) + " " + " " + " ", end ="")
            x_coordinates[i] = randomNumbers[i][0]
            y_coordinates[i] = randomNumbers[i][1]
            j += 1
        #print()
        i += 1

    #for j in range(0, len(randomNumbers)):   
    #print(x_coordinates, "\n", y_coordinates)
    plt.scatter(x_coordinates, y_coordinates, s=50, c= "orange")
    plt.show()

main([])
#print(x_coordinates, y_coordinates)
