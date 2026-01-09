import bpy
import os
import numpy as np

# Configuration
camera_name = "CM"
output_dir = "synthetic_data/rendered_data"
os.makedirs(output_dir, exist_ok=True)

# Get camera object
camera = bpy.data.objects.get(camera_name)
if not camera or camera.type != 'CAMERA':
    raise Exception("Camera not found or invalid")

scene_name = "Scene"  # Replace with the actual name of your scene
scene = bpy.data.scenes.get(scene_name)

# Setup compositor nodes for Depth and RGB
scene.use_nodes = True
nodes = scene.node_tree.nodes
links = scene.node_tree.links
nodes.clear()

render_layers = nodes.new('CompositorNodeRLayers')

# --- Depth Outputs ---
# EXR (raw float32 values)
depth_exr = nodes.new('CompositorNodeOutputFile')
depth_exr.base_path = output_dir
depth_exr.format.file_format = 'OPEN_EXR'
depth_exr.format.color_depth = '32'
depth_exr.file_slots[0].path = "depth_exr_"
links.new(render_layers.outputs['Depth'], depth_exr.inputs[0])

# PNG (raw camera-space values as 16-bit integers)
math_node = nodes.new('CompositorNodeMath')
math_node.operation = 'MULTIPLY'
math_node.inputs[1].default_value = 1000  # Scale factor to preserve mm precision

depth_png = nodes.new('CompositorNodeOutputFile')
links.new(render_layers.outputs['Depth'], math_node.inputs[0])
links.new(math_node.outputs[0], depth_png.inputs[0])

depth_png.base_path = output_dir
depth_png.format.file_format = 'PNG'
depth_png.format.color_depth = '16'
depth_png.file_slots[0].path = "depth_png_"

# --- RGB Output ---
rgb_output = nodes.new('CompositorNodeOutputFile')
rgb_output.base_path = output_dir
rgb_output.format.file_format = 'PNG'
rgb_output.file_slots[0].path = "rgb_"
links.new(render_layers.outputs['Image'], rgb_output.inputs[0])  

# Render the scene
bpy.ops.render.render(write_still=True)

# Camera parameters calculation
cam_data = camera.data
res_x = scene.render.resolution_x
res_y = scene.render.resolution_y

# Get shift values
shift_x = cam_data.shift_x
shift_y = cam_data.shift_y

# Correct focal length calculation based on sensor fit
sensor_fit = cam_data.sensor_fit
if sensor_fit == 'AUTO':
    sensor_fit = 'HORIZONTAL' if res_x >= res_y else 'VERTICAL'

if sensor_fit == 'HORIZONTAL':
    sensor_size = cam_data.sensor_width
    f_x = (cam_data.lens * res_x) / sensor_size
    f_y = f_x
else:
    sensor_size = cam_data.sensor_height
    f_y = (cam_data.lens * res_y) / sensor_size
    f_x = f_y

# Adjust principal point for shift
c_x = (0.5 - shift_x) * res_x
c_y = (0.5 - shift_y) * res_y

intrinsics = np.array([[f_x, 0, c_x],
                       [0, f_y, c_y],
                       [0, 0, 1]], dtype=np.float32)

# Correct extrinsics (world-to-camera matrix)
extrinsics = np.array(camera.matrix_world, dtype=np.float32)

# Save parameters
np.save(os.path.join(output_dir, "intrinsics.npy"), intrinsics)
np.save(os.path.join(output_dir, "extrinsics.npy"), extrinsics)

print(f"Data saved to: {output_dir}")

clip_params = np.array([cam_data.clip_start, cam_data.clip_end], dtype=np.float32)
np.save(os.path.join(output_dir, "clip_params.npy"), clip_params)