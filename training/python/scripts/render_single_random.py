import bpy
import os
import random
import time
import sys

if not 'ML_LIGHT_DIR' in os.environ:
    os.system("say %s" % ("Create root environment variable"))
    raise Exception("Create root environment variable")

root = os.environ['ML_LIGHT_DIR']

light = bpy.context.scene.objects.get('Light01')

argv_start = sys.argv.index('--') + 1

x = random.uniform(-3, 3)
y = random.uniform(-3, -0.5)
z = random.uniform(0.5, 3)
name = str(sys.argv[argv_start + 0])

light.location.x = x
light.location.y = y
light.location.z = z

timestamp = time.time()

bpy.context.scene.render.filepath = root + 'training/blender/render_single/point 80 W (5m)/{}'.format(name)
bpy.ops.render.render(write_still=True)