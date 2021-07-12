
import bpy
import random
import time
import os


DATA_BATCH_NAME = "ruslan 300 W uniform 320x320"
RESOLUTION_X = 320
RESOLUTION_Y = 320

scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION_X
scene.render.resolution_y = RESOLUTION_Y

if not 'ML_LIGHT_DIR' in os.environ:
    os.system("say %s" % ("Create root environment variable"))
    raise Exception("Create root environment variable")
    
root = os.environ['ML_LIGHT_DIR']

light = bpy.context.scene.objects.get('Light01')

step_x = 0.3
step_y = 0.15
step_z = -0.15

initial_x = -3
initial_y = -3.5
initial_z = 3.5

id = 0

x = initial_x
while(x <= 3):
    y = initial_y
    while(y <= -0.5):
        z = initial_z
        while(z >= 0.5):

            light.location.x = x
            light.location.y = y
            light.location.z = z

            timestamp = time.time()
            bpy.context.scene.render.filepath = root + \
            'training/blender/render/{}/{} {} {} {}'.format(DATA_BATCH_NAME, x, y, z, timestamp)
            bpy.ops.render.render(write_still=True)

            z += step_z
        y += step_y
    x += step_x


