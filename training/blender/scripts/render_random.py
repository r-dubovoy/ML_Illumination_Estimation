
import bpy
import random
import time
import os

if not 'ML_LIGHT_DIR' in os.environ:
    os.system("say %s" % ("Create root environment variable"))
    raise Exception("Create root environment variable")
    
root = os.environ['ML_LIGHT_DIR']

light = bpy.context.scene.objects.get('Light01')

for index in range(1, 500):
    
    x = random.uniform(-3, 3)
    y = random.uniform(-3, -0.5)
    z = random.uniform(0.5, 3)

    light.location.x = x
    light.location.y = y
    light.location.z = z
     
    timestamp = time.time()

    #timestamp will grow your directory, use index in 1..500 range
    bpy.context.scene.render.filepath = root + 'training/blender/render/point 80 W (5m)/{} {} {} {}'.format(x, y, z, timestamp)
    bpy.ops.render.render(write_still = True)