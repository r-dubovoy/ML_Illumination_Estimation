import bpy
import os
import random
import time
import sys

if not 'ML_LIGHT_DIR' in os.environ:
    os.system("say %s" % ("Create root environment variable"))
    raise Exception("Create root environment variable")

root = os.environ['ML_LIGHT_DIR']

hat_scene = bpy.data.scenes['Hat']
light_hat = hat_scene.objects.get('Light_hat')
light_hat.data.energy = 0

light_ruslan = bpy.context.scene.objects.get('Light_ruslan')
# light_hat.power = 0

argv_start = sys.argv.index('--') + 1

x = random.uniform(-3, 3)
y = random.uniform(-3, -0.5)
z = random.uniform(0.5, 3)
name = str(sys.argv[argv_start + 0])

light_hat.location.x = x
light_hat.location.y = y
light_hat.location.z = z


light_ruslan.location.x = x
light_ruslan.location.y = y
light_ruslan.location.z = z

timestamp = time.time()

bpy.context.scene.render.filepath = root + \
                                    'training/blender/render_single/layers_upd/' \
                                    '{} {} {} {}'.format(x, y, z, name)
bpy.ops.render.render(write_still=True)

