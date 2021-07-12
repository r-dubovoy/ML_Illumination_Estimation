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
light_hat.data.energy = 300

light_ruslan = bpy.context.scene.objects.get('Light_ruslan')
# light_hat.power = 0

argv_start = sys.argv.index('--') + 1

name = str(sys.argv[argv_start + 0])
print(name + '!!!!!!!!!!!!!!!!!')
x_r = float(sys.argv[argv_start + 1])
y_r = float(sys.argv[argv_start + 2])
z_r = float(sys.argv[argv_start + 3])

x_h = float(sys.argv[argv_start + 4])
y_h = float(sys.argv[argv_start + 5])
z_h = float(sys.argv[argv_start + 6])


light_hat.location.x = x_h
light_hat.location.y = y_h
light_hat.location.z = z_h


light_ruslan.location.x = x_r
light_ruslan.location.y = y_r
light_ruslan.location.z = z_r

timestamp = time.time()

bpy.context.scene.render.filepath = root + \
                                    'training/blender/render_single/layers_upd/' \
                                    '{} {} {} {}'.format(x_r, y_r, z_r, name)
bpy.ops.render.render(write_still=True)

