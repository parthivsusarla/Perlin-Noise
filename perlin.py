import noise
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

shape = (1000, 1000)
scale = 100.0
persistence = 0.5
lacunarity = 2.0
octaves = 6

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=0)

# Convert the generated noise array to a grayscale image
img = Image.fromarray((world * 255).astype('uint8'), mode='L')
img.save('gray_world.png')

# Define color values
blue = [48, 127, 230]  
green = [53, 161, 71]
sand = [222, 175, 98]
snow = [218, 237, 237]
mountain = [120, 130, 116]

def color(world):
    color_world = np.zeros(world.shape + (3,), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
             if world[i][j] < -0.05:
                color_world[i][j] = blue
             elif world[i][j] < 0:
                color_world[i][j] = sand
             elif world[i][j] < 0.3:
                color_world[i][j] = green
             elif world[i][j] < 0.37:
                color_world[i][j] = mountain
             else:
                color_world[i][j] = snow
    return color_world

colorworld = color(world)
img_color = Image.fromarray(colorworld, mode='RGB')
img_color.save('world.png')

# Define the island mask
a, b = shape[0] / 2, shape[1] / 2
n = 1024
r = 125
y, x = np.ogrid[-a:n-a, -b:n-b]
mask = x**2 + y**2 <= r**2

black = [0, 0, 0]
island = np.zeros_like(colorworld, dtype=np.uint8)

for i in range(shape[0]):
    for j in range(shape[1]):
        if mask[i][j]:
            island[i][j] = colorworld[i][j]
        else:
            island[i][j] = black

img_island = Image.fromarray(island, mode='RGB')
img_island.save('island.png')


# Set up parameters for generating 3D Perlin noise
shape = (100,100)  # Reducing the size for faster rendering
# Generate 3D Perlin noise
world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, 
                                    persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=0)

# Create a 3D grid
x, y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]))

# Define a function to map the grayscale values to height
def map_to_height(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

min_val = np.min(world)
max_val = np.max(world)

z = map_to_height(world, min_val, max_val)

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define color mapping for terrain
colors = plt.cm.terrain(z)  # Using the terrain colormap
#Normalize the colors correctly
norm = plt.Normalize(z.min(), z.max())

# Plot the 3D terrain with colors
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, linewidth=0, antialiased=False)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Height)')
# Add colorbar for elevation
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.terrain, norm=norm), ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Elevation')
# Show the plot
plt.show()





