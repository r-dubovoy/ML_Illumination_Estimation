import bpy
import os
import time
import sys

if not 'ML_LIGHT_DIR' in os.environ:
    os.system("say %s" % ("Create root environment variable"))
    raise Exception("Create root environment variable")

root = os.environ['ML_LIGHT_DIR']

light = bpy.context.scene.objects.get('Light01')

argv_start = sys.argv.index('--') + 1

x = float(sys.argv[argv_start + 0])
y = float(sys.argv[argv_start + 1])
z = float(sys.argv[argv_start + 2])
name = str(sys.argv[argv_start + 3])

light.location.x = x
light.location.y = y
light.location.z = z

timestamp = time.time()

# timestamp will grow your directory, use index in 1..500 range
bpy.context.scene.render.filepath = root + 'training/blender/render_single/point 80 W (5m)/{}'.format(name)
bpy.ops.render.render(write_still=True)