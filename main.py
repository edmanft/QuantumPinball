import numpy as np
import pygame
import sys

# Initialize Pygame
pygame.init()

# Physical and simulation constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Particle mass
a = 1.0     # Length of the box
nx = 800    # Number of spatial points
dx = a / (nx - 1)
x = np.linspace(0, a, nx)

# Time parameters
dt = 1e-5         # Time step
total_time = 0.002  # Total simulation time
steps = int(total_time / dt)

# Gaussian wave packet parameters
x0 = a / 4       # Initial center position
sigma = 0.05     # Width of the Gaussian
k0 = 200 * np.pi / a  # Initial wave number

# Initialize the wave function
psi = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
psi *= np.exp(1j * k0 * x)
psi[0] = psi[-1] = 0  # Enforce boundary conditions
# Normalize
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

# Laplacian operator
laplacian = -2 * np.eye(nx, k=0) + np.eye(nx, k=1) + np.eye(nx, k=-1)
laplacian /= dx ** 2
# Apply boundary conditions to the Laplacian
laplacian[0, :] = laplacian[-1, :] = 0
laplacian[:, 0] = laplacian[:, -1] = 0

# Hamiltonian operator
H = - (hbar ** 2) / (2 * m) * laplacian

# Crank-Nicolson matrices
I = np.eye(nx)
A = I + 1j * H * dt / (2 * hbar)
B = I - 1j * H * dt / (2 * hbar)
# Precompute the inverse of A
A_inv = np.linalg.inv(A)

# Pygame window settings
width, height = 800, 600  # Window size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Quantum Wave Packet Evolution')

# Scaling factors
x_scale = width / (a)
y_scale = height * 0.8  # Scale to 80% of window height
y_offset = height / 2   # Center vertically

# Color settings
background_color = (0, 0, 0)  # Black
wave_color = (0, 255, 255)    # Cyan

# Font settings
font = pygame.font.SysFont('Arial', 20)

# Time evolution function
def time_evolve(psi, A_inv, B):
    psi = A_inv @ (B @ psi)
    # Enforce boundary conditions
    psi[0] = psi[-1] = 0
    return psi

# Main simulation loop
running = True
clock = pygame.time.Clock()
step = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    if step >= steps:
        running = False
        break

    # Time evolution
    psi = time_evolve(psi, A_inv, B)
    # Normalize
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    # Clear the screen
    screen.fill(background_color)

    # Draw the probability density
    prob_density = np.abs(psi) ** 2
    max_prob = np.max(prob_density)
    for i in range(nx - 1):
        x1 = int(x[i] * x_scale)
        x2 = int(x[i+1] * x_scale)
        y1 = int(y_offset - prob_density[i] / max_prob * y_scale)
        y2 = int(y_offset - prob_density[i+1] / max_prob * y_scale)
        pygame.draw.line(screen, wave_color, (x1, y1), (x2, y2))

    # Display time
    time_text = font.render(f'Time: {step * dt:.5f} s', True, (255, 255, 255))
    screen.blit(time_text, (10, 10))

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(60)  # Limit to 60 FPS

    step += 1

# Quit Pygame
pygame.quit()
sys.exit()
