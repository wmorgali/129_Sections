import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
from math import atan2

# Set mesh to 'mesh.dat' in the same directory
mesh = 'mesh.dat'  # Automatically looks for 'mesh.dat' in the current directory

# Read the data from the file
data = pd.read_csv(mesh, delim_whitespace=True)

x = data['X']
y = data['Y']

# Plot the data points and save them to mesh.png
plt.scatter(x, y, color='blue', label='Data Points')
plt.title("Data Points from mesh.dat")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.savefig('mesh.png', dpi=300)  # Save as a high-quality image (300 DPI)
plt.show()

# Convert the data to a list of points
pts = data[['X', 'Y']].values.tolist()

# Convex Hull
# Creates a scatter plot of the points and the convex hull saved to grahamscan.png
def scatter_plot(coords, convex_hull=None, filename='grahamscan.png'):
    xs, ys = zip(*coords)
    plt.scatter(xs, ys, color='blue', label='Data Points')
    if convex_hull is not None:
        for i in range(1, len(convex_hull) + 1):
            if i == len(convex_hull): i = 0
            c0 = convex_hull[i-1]
            c1 = convex_hull[i]
            plt.plot((c0[0], c1[0]), (c0[1], c1[1]), 'r', label='Convex Hull' if i == 1 else "")
        plt.savefig(filename, dpi=300)  # Save as a high-quality image (300 DPI)
    plt.title("Convex Hull")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Polar Angle between 2 points
def polar_angle(p0, p1):
    y_span = p0[1] - p1[1]
    x_span = p0[0] - p1[0]
    return atan2(y_span, x_span)

# Distance between 2 points
def distance(p0, p1):
    y_span = p0[1] - p1[1]
    x_span = p0[0] - p1[0]
    return y_span**2 + x_span**2

# Determinant of 3 points
def det(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

# Quicksort
def quicksort(a, anchor):
    if len(a) <= 1: return a
    smaller, equal, larger = [], [], []
    piv_ang = polar_angle(a[randint(0, len(a) - 1)], anchor)
    for pt in a:
        pt_ang = polar_angle(pt, anchor)
        if pt_ang < piv_ang:  smaller.append(pt)
        elif pt_ang == piv_ang: equal.append(pt)
        else: larger.append(pt)
    return quicksort(smaller, anchor) + sorted(equal, key=lambda x: distance(x, anchor)) + quicksort(larger, anchor)

# Graham Scan function
def graham_scan(points, show_progress=False):
    points = [tuple(p) for p in points]  # Convert to tuples
    min_idx = None
    for i, (x, y) in enumerate(points):
        if min_idx is None or y < points[min_idx][1]:
            min_idx = i
        if y == points[min_idx][1] and x < points[min_idx][0]:
            min_idx = i
    anchor = points[min_idx]
    sorted_pts = quicksort(points, anchor)
    sorted_pts = [tuple(p) for p in sorted_pts]  # Ensure tuples for indexing
    anchor = tuple(anchor)
    sorted_pts.remove(anchor)  # Remove anchor safely
    hull = [anchor, sorted_pts[0]]
    for s in sorted_pts[1:]:
        while det(hull[-2], hull[-1], s) <= 0:
            del hull[-1]
            if len(hull) < 2: break
        hull.append(s)
        if show_progress: scatter_plot(points, hull)
    return hull


# Monotone Chain Algorithm
def monotone_chain(points):
    # Sort the points lexographically (first by x, then by y)
    points = sorted(points, key=lambda p: (p[0], p[1]))

    # Build the lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and det(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and det(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate the lower and upper hulls, removing duplicates
    convex_hull = lower[:-1] + upper[:-1]
    return convex_hull

# QuickHull Algorithm
def quickhull(points):
    # convert points to a list of tuples for consistency
    points = [tuple(p) for p in points]
    if len(points) < 3:
        return points

    # helper: which side of line from a to b is point p on?
    def side(a, b, p):
        return det(a, b, p)

    # recursively add hull points between p and q
    def add_hull(pt_set, p, q):
        index = None
        max_distance = 0
        # find the point with maximum distance from line p->q
        for i, pt in enumerate(pt_set):
            d = abs(det(p, q, pt))
            if d > max_distance:
                max_distance = d
                index = i
        if index is None:
            # no point is outside; p->q is a hull edge
            return []
        farthest = pt_set[index]
        # split set into two subsets: points to the left of (p, farthest)
        # and points to the left of (farthest, q)
        left_set = [pt for pt in pt_set if side(p, farthest, pt) > 0]
        right_set = [pt for pt in pt_set if side(farthest, q, pt) > 0]
        # recursively find hull points on these segments
        return add_hull(left_set, p, farthest) + [farthest] + add_hull(right_set, farthest, q)

    # find the leftmost and rightmost points (extremes)
    leftmost = min(points, key=lambda p: p[0])
    rightmost = max(points, key=lambda p: p[0])

    # partition the points into two subsets: above and below the line
    above = [pt for pt in points if side(leftmost, rightmost, pt) > 0]
    below = [pt for pt in points if side(rightmost, leftmost, pt) > 0]

    # build the hull (in order) by combining the extreme points and the recursively found points
    upper_hull = add_hull(above, leftmost, rightmost)
    lower_hull = add_hull(below, rightmost, leftmost)

    # The full hull is the leftmost point, then the points on the upper hull,
    # then the rightmost point, and finally the points on the lower hull.
    return [leftmost] + upper_hull + [rightmost] + lower_hull

# Jarvis Algorithm (Gift Wrapping)
def jarvis(points):
    n = len(points)
    hull = []
    leftmost_idx = np.argmin(points[:, 0])
    current_idx = leftmost_idx

    while True:
        hull.append(tuple(points[current_idx]))
        next_idx = (current_idx + 1) % n
        
        for i in range(n):
            if det(points[current_idx], points[next_idx], points[i]) > 0:
                next_idx = i

        current_idx = next_idx

        if current_idx == leftmost_idx:
            break

    return hull

# Convert points to numpy array for Jarvis
points_np = np.array(pts)

# Run the Jarvis Algorithm and plot the convex hull
convex_hull_jarvis = jarvis(points_np)
scatter_plot(pts, convex_hull_jarvis, filename='jarvis.png')

# Run the QuickHull Algorithm and plot the convex hull
convex_hull_quickhull = quickhull(pts)
scatter_plot(pts, convex_hull_quickhull, filename='quickhull.png')

# Run the Graham Scan and plot the convex hull
convex_hull_graham = graham_scan(pts, False)
scatter_plot(pts, convex_hull_graham, filename='grahamscan.png')

# Run the Monotone Chain Algorithm and plot the convex hull
convex_hull_monotone = monotone_chain(pts)
scatter_plot(pts, convex_hull_monotone, filename='monotone.png')

# Plot all 4 hulls on the mesh scatter plot and save as "hulls.png"
plt.scatter(x, y, color='blue', label='Data Points')
for hull, label, color in [
    (convex_hull_quickhull, 'QuickHull', 'r'),
    (convex_hull_graham, 'Graham Scan', 'g'),
    (convex_hull_monotone, 'Monotone', 'y'),
    (convex_hull_jarvis, 'Jarvis', 'b')]:
    for i in range(1, len(hull) + 1):
        if i == len(hull): i = 0
        c0 = hull[i-1]
        c1 = hull[i]
        plt.plot((c0[0], c1[0]), (c0[1], c1[1]), color, label=label if i == 1 else "")
plt.title("All Convex Hulls")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.savefig('hulls.png', dpi=300)
plt.show()


# Part 2:

import numpy as np
import time
import matplotlib.pyplot as plt

# Function to generate a uniform 2D point cloud
def generate_uniform_point_cloud(n, bounds=(-1, 1)):
    np.random.seed(42)
    return np.random.uniform(bounds[0], bounds[1], (n, 2))

# Function to time convex hull algorithms
def time_algorithm(algorithm, points):
    start_time = time.time()
    algorithm(points)
    return time.time() - start_time

# List of point counts to test
n_values = [10, 50, 100, 200, 400, 800, 1000]

# Store results
results = {"Graham Scan": [], "Jarvis March": [], "QuickHull": [], "Monotone Chain": []}

# Run experiments
for n in n_values:
    points = generate_uniform_point_cloud(n)
    results["Graham Scan"].append(time_algorithm(graham_scan, points))
    results["Jarvis March"].append(time_algorithm(jarvis, points))
    results["QuickHull"].append(time_algorithm(quickhull, points))
    results["Monotone Chain"].append(time_algorithm(monotone_chain, points))

# Plot results
plt.figure(figsize=(8, 6))
for algo, times in results.items():
    plt.plot(n_values, times, label=algo, marker='o')

plt.xlabel("Number of Points (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Time Complexity of Convex Hull Algorithms")
plt.legend()
plt.grid(True)
plt.savefig("time_complexity.png", dpi=300)
plt.show()

# Save conclusion to a text file
conclusion = """Jarvis takes the longest. The rest are roughly the same.
"""
with open("runtime_conclusion.txt", "w") as f:
    f.write(conclusion)



# Function to generate a uniform 2D point cloud
def generate_uniform_point_cloud5(n, bounds=(-5, 5)):
    np.random.seed(42)
    return np.random.uniform(bounds[0], bounds[1], (n, 2))

# Function to time convex hull algorithms
def time_algorithm(algorithm, points):
    start_time = time.time()
    algorithm(points)
    return time.time() - start_time

# List of point counts to test
n_values = [10, 50, 100, 200, 400, 800, 1000]

# Store results
results = {"Graham Scan": [], "Jarvis March": [], "QuickHull": [], "Monotone Chain": []}

# Run experiments
for n in n_values:
    points = generate_uniform_point_cloud5(n)
    results["Graham Scan"].append(time_algorithm(graham_scan, points))
    results["Jarvis March"].append(time_algorithm(jarvis, points))
    results["QuickHull"].append(time_algorithm(quickhull, points))
    results["Monotone Chain"].append(time_algorithm(monotone_chain, points))

# Plot results
plt.figure(figsize=(8, 6))
for algo, times in results.items():
    plt.plot(n_values, times, label=algo, marker='o')

plt.xlabel("Number of Points (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Time Complexity of Convex Hull Algorithms")
plt.legend()
plt.grid(True)
plt.savefig("time_complexity5bounds.png", dpi=300)
plt.show()

# Save conclusion to a text file
conclusion5 = """The run times stay roughly the same.
"""
with open("runtime_conclusion5.txt", "w") as f:
    f.write(conclusion5)


# Function to generate a Gaussian-distributed 2D point cloud
def generate_gaussian_point_cloud(n, mean=0, std_dev=1):
    np.random.seed(42)
    return np.random.normal(mean, std_dev, (n, 2))

# Function to time convex hull algorithms
def time_algorithm(algorithm, points):
    start_time = time.time()
    algorithm(points)
    return time.time() - start_time

# List of point counts to test
n_values = [10, 50, 100, 200, 400, 800, 1000]

# Store results for Gaussian distribution
results_gaussian = {"Graham Scan": [], "Jarvis March": [], "QuickHull": [], "Monotone Chain": []}

# Run experiments for Gaussian distribution
for n in n_values:
    points = generate_gaussian_point_cloud(n)
    results_gaussian["Graham Scan"].append(time_algorithm(graham_scan, points))
    results_gaussian["Jarvis March"].append(time_algorithm(jarvis, points))
    results_gaussian["QuickHull"].append(time_algorithm(quickhull, points))
    results_gaussian["Monotone Chain"].append(time_algorithm(monotone_chain, points))

# Plot results for Gaussian distribution
plt.figure(figsize=(8, 6))
for algo, times in results_gaussian.items():
    plt.plot(n_values, times, label=algo, marker='o')

plt.xlabel("Number of Points (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Time Complexity of Convex Hull Algorithms (Gaussian Distribution)")
plt.legend()
plt.grid(True)
plt.savefig("time_complexity_gaussian.png", dpi=300)
plt.show()

# Save Gaussian distribution conclusion to a text file
conclusion_gaussian = """The runtime does not significantly change due to the variance in the Gaussian distribution."""
with open("runtime_conclusion_gaussian.txt", "w") as f:
    f.write(conclusion_gaussian)

# Part d: Run multiple sets of point clouds for n=50 and analyze runtime distributions
n_runs = 100
runtime_distributions = {"Graham Scan": [], "Jarvis March": [], "QuickHull": [], "Monotone Chain": []}

for _ in range(n_runs):
    points = generate_uniform_point_cloud(50)
    runtime_distributions["Graham Scan"].append(time_algorithm(graham_scan, points))
    runtime_distributions["Jarvis March"].append(time_algorithm(jarvis, points))
    runtime_distributions["QuickHull"].append(time_algorithm(quickhull, points))
    runtime_distributions["Monotone Chain"].append(time_algorithm(monotone_chain, points))

# Plot histograms for runtime distributions
plt.figure(figsize=(12, 8))
for i, (algo, runtimes) in enumerate(runtime_distributions.items(), 1):
    plt.subplot(2, 2, i)
    plt.hist(runtimes, bins=10, alpha=0.7, label=algo, color=np.random.rand(3,))
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Frequency")
    plt.title(f"Runtime Distribution of {algo}")
    plt.legend()
plt.tight_layout()
plt.savefig("runtime_distributions.png", dpi=300)
plt.show()

# Save conclusion for runtime distribution analysis
conclusion_runtime_dist = """The runtime distribution varies among algorithms, with Jarvis March showing the highest variability.
Graham Scan and QuickHull exhibit relatively stable performance, while Monotone Chain shows moderate fluctuations."""
with open("runtime_conclusion_distribution.txt", "w") as f:
    f.write(conclusion_runtime_dist)
