import bpy
import mathutils
import math

# Get the 3D cursor location
cursor_location = bpy.context.scene.cursor.location

#######################################################

# Get the camera parameter
camera_name = "CM" 
plane_name = "main door middle.001"           # Mirror plane
mirrored_camera_name = "Mirrored_CM"  # Mirrored camera name

#######################################################

camera = bpy.data.objects.get(camera_name)
plane = bpy.data.objects.get(plane_name)

if not camera or camera.type != 'CAMERA':
    print(f"Camera '{camera_name}' not found or is not a camera.")
elif not plane or plane.type != 'MESH':
    print(f"Plane '{plane_name}' not found or is not a mesh.")
else:
    # Get plane world matrix and normal
    plane_matrix = plane.matrix_world
    plane_normal = plane_matrix.to_3x3() @ mathutils.Vector((0, 0, 1))
    plane_point = plane_matrix.translation
    plane_normal.normalize()

    # Create a reflection matrix
    n = plane_normal
    reflection_matrix = mathutils.Matrix((
        (1 - 2 * n.x**2, -2 * n.x * n.y, -2 * n.x * n.z, 0),
        (-2 * n.x * n.y, 1 - 2 * n.y**2, -2 * n.y * n.z, 0),
        (-2 * n.x * n.z, -2 * n.y * n.z, 1 - 2 * n.z**2, 0),
        (0, 0, 0, 1)
    ))

    # Get camera transform
    cam_matrix = camera.matrix_world
    cam_local = cam_matrix.translation - plane_point

    # Compute mirrored transform
    mirrored_location = reflection_matrix @ cam_local + plane_point
    mirrored_rotation = reflection_matrix.to_3x3() @ cam_matrix.to_3x3()

    flip_x = mathutils.Matrix((
        (-1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ))
    mirrored_rotation = (mirrored_rotation @ flip_x).to_4x4()

    # Create or reuse mirrored camera
    mirrored_camera = bpy.data.objects.get(mirrored_camera_name)
    if not mirrored_camera:
        mirrored_camera = bpy.data.objects.new(mirrored_camera_name, camera.data.copy())
        # Add mirrored camera to the same collection as the original camera
        for col in camera.users_collection:
            col.objects.link(mirrored_camera)

    # Set mirrored camera transformation
    mirrored_camera.matrix_world = mathutils.Matrix.Translation(mirrored_location) @ mirrored_rotation.to_4x4()

    # Copy lens settings from the original camera
    mirrored_camera.data.lens = camera.data.lens
    mirrored_camera.data.sensor_width = camera.data.sensor_width
    mirrored_camera.data.sensor_height = camera.data.sensor_height

    # Select cameras for easy comparison
    bpy.context.view_layer.objects.active = mirrored_camera
    mirrored_camera.select_set(True)
    camera.select_set(True)

    # Print results
    print(f"Original Camera Location: {camera.location}")
    print(f"Mirrored Camera Location: {mirrored_camera.location}")
    print(f"Mirrored Camera Rotation (Matrix):\n{mirrored_rotation}")
    print("Mirrored camera with correct rotation added to the viewport.")


