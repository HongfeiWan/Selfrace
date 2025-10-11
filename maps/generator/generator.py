import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, NamedTuple


class Waypoint(NamedTuple):
    """A waypoint with position, direction, and unique ID information."""
    waypoint_id: int
    x: float
    y: float
    heading: float  # Direction in radians (0 = East, π/2 = North)

class RoadNet:
    """
    A class for generating road network waypoints.
    """
    def __init__(self):
        """Initialize the RoadNet class."""
        pass
    def generate_circular_multi_lane_waypoints(self, 
                                            center: Tuple[float, float] = (0.0, 0.0),
                                            inner_radius: float = 50.0,
                                            num_lanes: int = 2,
                                            lane_spacing: float = 4.0,
                                            num_points_per_lane: int = None) -> Dict[str, List[Waypoint]]:
        """
        Generate circular multi-lane waypoints with direction information.
        
        Args:
            center: Center point of the circle (x, y)
            inner_radius: Radius of the innermost circle
            num_lanes: Number of concentric lanes to generate
            lane_spacing: Distance between adjacent lanes (meters)
            num_points_per_lane: Number of points per lane. If None, calculated based on circumference
        
        Returns:
            Dictionary containing lane waypoints with direction, keys are 'lane_0', 'lane_1', etc.
        """
        center_x, center_y = center
        
        # Calculate circumference and number of points if not specified
        if num_points_per_lane is None:
            circumference = 2 * math.pi * inner_radius
            num_points_per_lane = int(circumference / 1.0)  # 1m Frenet distance between points
        
        # Generate angles for waypoints
        angles = np.linspace(0, 2 * math.pi, num_points_per_lane, endpoint=False)
        
        # Generate waypoints for each lane
        all_lanes = {}
        waypoint_id_counter = 0
        
        for lane_idx in range(num_lanes):
            # Calculate radius for this lane
            current_radius = inner_radius + lane_idx * lane_spacing
            
            # Generate waypoints for this lane
            lane_waypoints = []
            for i, angle in enumerate(angles):
                x = center_x + current_radius * math.cos(angle)
                y = center_y + current_radius * math.sin(angle)
                # Heading is tangent to the circle (perpendicular to radius)
                # For counter-clockwise direction: heading = angle + π/2
                heading = angle + math.pi / 2
                waypoint_id = waypoint_id_counter
                waypoint_id_counter += 1
                lane_waypoints.append(Waypoint(waypoint_id, x, y, heading))
            
            # Store lane waypoints
            all_lanes[f'lane_{lane_idx}'] = lane_waypoints
        
        return all_lanes
    
    def calculate_frenet_distance(self, point1: Waypoint, 
                                 point2: Waypoint, 
                                 center: Tuple[float, float] = (0.0, 0.0)) -> float:
        """
        Calculate Frenet distance between two points on a circular path.
        
        Args:
            point1: First waypoint
            point2: Second waypoint
            center: Center of the circle (x, y)
        
        Returns:
            Frenet distance (arc length) between the points
        """
        center_x, center_y = center
        
        # Calculate angles from center
        angle1 = math.atan2(point1.y - center_y, point1.x - center_x)
        angle2 = math.atan2(point2.y - center_y, point2.x - center_x)
        
        # Calculate angular difference
        angle_diff = abs(angle2 - angle1)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # Calculate radius (average of both points)
        radius1 = math.sqrt((point1.x - center_x)**2 + (point1.y - center_y)**2)
        radius2 = math.sqrt((point2.x - center_x)**2 + (point2.y - center_y)**2)
        avg_radius = (radius1 + radius2) / 2
        
        # Frenet distance is arc length
        return avg_radius * angle_diff
    
    def calculate_euclidean_distance(self, point1: Waypoint, 
                                   point2: Waypoint) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First waypoint
            point2: Second waypoint
        
        Returns:
            Euclidean distance between the points
        """
        return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)
    
    def visualize_waypoints(self, waypoints: Dict[str, List[Waypoint]], 
                          center: Tuple[float, float] = (0.0, 0.0),
                          title: str = "Circular Multi-Lane Waypoints",
                          save_path: str = None) -> None:
        """
        Visualize the generated waypoints.
        
        Args:
            waypoints: Dictionary containing lane waypoints
            center: Center of the circle
            title: Title for the plot
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Define colors for different lanes
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Plot all lanes
        for lane_key, lane_waypoints in waypoints.items():
            if lane_key.startswith('lane_'):
                lane_idx = int(lane_key.split('_')[1])
                color = colors[lane_idx % len(colors)]
                x = [point.x for point in lane_waypoints]
                y = [point.y for point in lane_waypoints]
                ax.scatter(x, y, c=color, s=5, alpha=0.8, label=f'Lane {lane_idx}')
        
        # Plot center point
        ax.plot(center[0], center[1], 'ko', markersize=8, label='Center')
        
        # Add arrows to show direction using heading information
        for lane_key, lane_waypoints in waypoints.items():
            if lane_key.startswith('lane_') and len(lane_waypoints) > 0:
                lane_idx = int(lane_key.split('_')[1])
                color = colors[lane_idx % len(colors)]
                
                # Add direction arrows for this lane
                for i in range(0, len(lane_waypoints), max(1, len(lane_waypoints)//8)):
                    heading = lane_waypoints[i].heading
                    dx = math.cos(heading) * 3  # 3m arrow length
                    dy = math.sin(heading) * 3
                    ax.arrow(lane_waypoints[i].x, lane_waypoints[i].y, dx, dy, 
                            head_width=1, head_length=1, fc=color, ec=color, alpha=0.8)
        
        # Calculate and display some statistics
        if waypoints:
            # Get first lane for statistics
            first_lane_key = next(key for key in waypoints.keys() if key.startswith('lane_'))
            first_lane = waypoints[first_lane_key]
            
            if len(first_lane) > 1:
                # Calculate average Frenet distance for first lane
                frenet_distances = []
                for i in range(len(first_lane) - 1):
                    dist = self.calculate_frenet_distance(
                        first_lane[i], first_lane[i+1], center
                    )
                    frenet_distances.append(dist)
                avg_frenet = np.mean(frenet_distances)
                
                # Calculate average lane distance between adjacent lanes
                lane_distances = []
                lane_keys = [key for key in waypoints.keys() if key.startswith('lane_')]
                lane_keys.sort(key=lambda x: int(x.split('_')[1]))
                
                for i in range(len(lane_keys) - 1):
                    current_lane = waypoints[lane_keys[i]]
                    next_lane = waypoints[lane_keys[i + 1]]
                    for j in range(len(current_lane)):
                        dist = self.calculate_euclidean_distance(
                            current_lane[j], next_lane[j]
                        )
                        lane_distances.append(dist)
                avg_lane = np.mean(lane_distances) if lane_distances else 0
                
                # Add text box with statistics
                stats_text = f'Number of lanes: {len(lane_keys)}\n'
                stats_text += f'Points per lane: {len(first_lane)}\n'
                for lane_key in lane_keys:
                    lane_idx = int(lane_key.split('_')[1])
                    lane_waypoints = waypoints[lane_key]
                    if lane_waypoints:
                        ids = [wp.waypoint_id for wp in lane_waypoints]
                        stats_text += f'Lane {lane_idx} IDs: {min(ids)}-{max(ids)}\n'
                stats_text += f'Avg Frenet distance: {avg_frenet:.2f}m\n'
                stats_text += f'Avg lane distance: {avg_lane:.2f}m'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8))
        
        # Set equal aspect ratio and grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Set reasonable limits
        all_x = []
        all_y = []
        for lane_waypoints in waypoints.values():
            if lane_waypoints:
                all_x.extend([wp.x for wp in lane_waypoints])
                all_y.extend([wp.y for wp in lane_waypoints])
        
        if all_x and all_y:
            margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.1
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create RoadNet instance
    road_net = RoadNet()
    # Generate circular multi-lane waypoints
    waypoints = road_net.generate_circular_multi_lane_waypoints(
        center=(0.0, 0.0),
        inner_radius=50.0,
        num_lanes=2,  # Generate 3 concentric lanes
        lane_spacing=4.0
    )
    
    print(f"Generated {len(waypoints)} lanes")
    for lane_key, lane_waypoints in waypoints.items():
        if lane_waypoints:
            lane_idx = int(lane_key.split('_')[1])
            print(f"Lane {lane_idx}: {len(lane_waypoints)} points, first point: ID={lane_waypoints[0].waypoint_id}, ({lane_waypoints[0].x:.2f}, {lane_waypoints[0].y:.2f}), heading: {lane_waypoints[0].heading:.2f} rad")
    
    # Visualize the waypoints
    print("\nVisualizing waypoints...")
    road_net.visualize_waypoints(waypoints, 
                                center=(0.0, 0.0),
                                title="Circular Multi-Lane Waypoints",
                                save_path="circular_multi_lane_waypoints.png")
