import os
import json
from tqdm import tqdm
from fluid import StableFluids
from examples import get_example_setup
from utils import parse_args, frames2gif

# read config
cfg = parse_args()
output_dir = f"outputs/{cfg['example']}"
os.makedirs(output_dir, exist_ok=True)
draw_dir = os.path.join(output_dir, cfg["draw"])
os.makedirs(draw_dir, exist_ok=True)
path = os.path.join(output_dir, "saved_config.json")
with open(path, "w") as fp:
    json.dump(cfg, fp, indent=2)

# get example setup
setup = get_example_setup(cfg["example"])

# init
fluid = StableFluids(cfg["N"], cfg["dt"], setup["domain"], cfg["visc"], cfg["diff"], setup["boundary_func"])

# set initial condition
fluid.add_source("velocity", setup["vsource"])
fluid.add_source("density", setup["dsource"])

# run simulation and draw frames
fluid.draw(cfg["draw"], os.path.join(draw_dir, f"{0:04d}.png"))
for i in tqdm(range(1, cfg["T"] + 1), desc="Simulate and draw"):
    fluid.step()

    fluid.draw(cfg["draw"], os.path.join(draw_dir, f"{i:04d}.png"))

    if i < setup["src_duration"]:
        fluid.add_source("velocity", setup["vsource"])
        fluid.add_source("density", setup["dsource"])

# frames to gif animation
save_path = os.path.join(os.path.dirname(draw_dir), f"anim_{cfg['draw']}.gif")
frames2gif(draw_dir, save_path, cfg["fps"])
