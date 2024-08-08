import numpy as np
import matplotlib.pyplot as plt

# Define the size of the chessboard
board_size = 8

# Create an 8x8 grid initialized with zeros
chessboard = np.zeros((board_size, board_size))

 # Fill the grid with black and white squares
# We use mod operation to alternate colors
for row in range(board_size):
    for col in range(board_size):
        if (row + col) % 2 == 0:
            chessboard[row, col] = 1  # White square
        else:
            chessboard[row, col] = 0  # Black square

# Display the chessboard using matplotlib
plt.imshow(chessboard, cmap='gray', interpolation='nearest')
plt.axis('off')  # Hide the axes

# Save the image as a PNG file
plt.savefig('chessboard.png', bbox_inches='tight', pad_inches=0, dpi=300)

#now read the same image and save image in csv using pandas
import pandas as pd
import cv2
#chessboard.png
img = cv2.imread('chessboard.png',0)
df = pd.DataFrame(img)
df.to_csv('chessboard.csv',index=False)
#now read the csv and save image
df = pd.read_csv('chessboard.csv')
img = df.values
cv2.imwrite('chessboard2.png',img)


