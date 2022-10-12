import sys
from pathlib import Path

from pymol import cmd

_, folder = sys.argv
# Load
cmd.loadall(f"{folder}/*.pdb")
cmd.bg_color("white")
cmd.remove("solvent")

# Settings
cmd.set("cartoon_fancy_helices", 1)
cmd.set("antialias", 1)
cmd.set("specular", 0)
cmd.set("cartoon_sampling", 30)
cmd.set("dot_width", 2)
cmd.set("dot_density", 1)
cmd.set("direct", 1.0)
cmd.set("ribbon_radius", 0.2)
cmd.set("cartoon_color", "0x775abf")
cmd.set("dot_color", "lightblue")
cmd.set("sphere_color", "lightblue")
cmd.set("cartoon_highlight_color", "lightblue")
cmd.set("sphere_scale", 0.5)
cmd.set("sphere_mode", 3)

# Make grid
cmd.select("rm", "br. b=0")
cmd.remove("rm")
objects = cmd.get_object_list(selection="(all)")
selection_string = []

for o in objects:
    with open(Path(folder) / f"{o}.txt") as f:
        selection_string.append(f.read())
cmd.hide("cartoon")
cmd.select("calpha_kmer", " or ".join(selection_string))
cmd.show(representation="cartoon", selection=f"calpha_kmer")

cmd.select("calpha", "name CA")
cmd.zoom("calpha", complete=1)

selection_string = []
for o in objects:
    center = int(o.split("_")[-1])
    selection_string.append(f"({o} and name CA and not resi {center - 8}:{center+8+1})")
cmd.select("calpha_radius", " or ".join(selection_string))
cmd.show(representation="spheres", selection=f"calpha_radius")

cmd.set("grid_mode", 1)

# Ray trace settings
cmd.set("ray_trace_mode", 1)
cmd.set("ray_trace_fog", 0)
cmd.set("ray_opaque_background", "off")
cmd.set("ray_trace_gain", 0)
cmd.set("ray_trace_disco_factor", 1)
cmd.ray(1000)

# Save
filename = f"{folder}_grid.png"
cmd.png(str(filename), width=1000, height=1000, dpi=300)
cmd.quit()
