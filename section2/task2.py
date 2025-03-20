import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Part a:

def bottom_surface(x, y):
    """Defines a parabolic bottom surface."""
    return x**2 + y**2

def top_surface(x, y):
    """Defines an exponential top surface."""
    return np.exp(-x**2 - y**2)

def compute_boundary_radius():
    """Finds the boundary radius where the two surfaces meet."""
    equation = lambda r: r**2 - np.exp(-r**2)
    return fsolve(equation, 0.7)[0]

def generate_grid(R, num_points=50):
    """Generates grid points inside a circular boundary."""
    x = np.linspace(-R, R, num_points)
    y = np.linspace(-R, R, num_points)
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.flatten(), yv.flatten()
    mask = xv**2 + yv**2 <= R**2
    return np.vstack([xv[mask], yv[mask]]).T

def create_surface_cloud(grid_pts):
    """Creates point clouds for top and bottom surfaces."""
    z_upper = top_surface(grid_pts[:, 0], grid_pts[:, 1])
    z_lower = bottom_surface(grid_pts[:, 0], grid_pts[:, 1])
    return np.hstack([grid_pts, z_upper[:, None]]), np.hstack([grid_pts, z_lower[:, None]])


# Part b: Surface Triangulation

def delauneytriangle():
    R = compute_boundary_radius()
    grid_pts = generate_grid(R, num_points=50)
    upper_pts, lower_pts = create_surface_cloud(grid_pts)
    num_pts = grid_pts.shape[0]
    
    tri_2d = Delaunay(grid_pts)
    
    boundary_pts = [i for i, pt in enumerate(grid_pts) if np.isclose(np.linalg.norm(pt), R, atol=1e-3)]
    interior_pts = [i for i in range(num_pts) if i not in boundary_pts]
    
    vertex_top = upper_pts
    vertex_bottom = lower_pts[interior_pts]
    merged_vertices = np.vstack((vertex_top, vertex_bottom))
    
    index_map = {j: num_pts + i for i, j in enumerate(interior_pts)}
    index_map.update({i: i for i in boundary_pts})
    
    bottom_tri = np.array([[index_map[i] for i in tri] for tri in tri_2d.simplices])
    all_triangles = np.vstack((tri_2d.simplices, bottom_tri))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(merged_vertices[:, 0], merged_vertices[:, 1],
                    merged_vertices[:, 2], triangles=all_triangles,
                    cmap='coolwarm', edgecolor='none', alpha=0.8)
    plt.savefig("delauneytriangle.png")


# Part c:

def volume_triangulation():
    R = compute_boundary_radius()
    xy_pts = generate_grid(R, num_points=15)
    volume_pts = []
    num_levels = 4
    
    for pt in xy_pts:
        x, y = pt
        z_low = bottom_surface(x, y)
        z_high = top_surface(x, y)
        z_vals = np.linspace(z_low, z_high, num_levels)
        for z in z_vals:
            volume_pts.append([x, y, z])
    
    volume_pts = np.array(volume_pts)
    tri_3d = Delaunay(volume_pts)
    tetrahedra = tri_3d.simplices
    
    faces = {}
    for tet in tetrahedra:
        for face in [(tet[0], tet[1], tet[2]),
                     (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[3]),
                     (tet[1], tet[2], tet[3])]:
            face_sorted = tuple(sorted(face))
            faces[face_sorted] = faces.get(face_sorted, 0) + 1
    
    boundary_faces = [face for face, count in faces.items() if count == 1]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection([volume_pts[list(face)] for face in boundary_faces],
                             facecolor='cyan', edgecolor='gray', alpha=0.5)
    ax.add_collection3d(poly)
    plt.savefig("volumemesh.png")

# ----------------------------
# Part d: Surface Mesh from Volume Mesh
# ----------------------------

def extract_surface_from_volume():
    R = compute_boundary_radius()
    xy_pts = generate_grid(R, num_points=15)
    volume_pts = []
    num_levels = 4
    
    for pt in xy_pts:
        x, y = pt
        z_low = bottom_surface(x, y)
        z_high = top_surface(x, y)
        z_vals = np.linspace(z_low, z_high, num_levels)
        for z in z_vals:
            volume_pts.append([x, y, z])
    
    volume_pts = np.array(volume_pts)
    tri_3d = Delaunay(volume_pts)
    tetrahedra = tri_3d.simplices
    
    faces = {}
    for tet in tetrahedra:
        for face in [(tet[0], tet[1], tet[2]),
                     (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[3]),
                     (tet[1], tet[2], tet[3])]:
            face_sorted = tuple(sorted(face))
            faces[face_sorted] = faces.get(face_sorted, 0) + 1
    
    boundary_faces = np.array([face for face, count in faces.items() if count == 1])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(volume_pts[:, 0], volume_pts[:, 1], volume_pts[:, 2],
                    triangles=boundary_faces, cmap='plasma', edgecolor='none', alpha=0.8)
    plt.savefig("surfacefromvolume.png")

if __name__ == '__main__':
    delauneytriangle()
    volume_triangulation()
    extract_surface_from_volume()
