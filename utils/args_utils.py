import argparse
import json


def parse_args() -> dict:
    """parse command-line arguments.

    Returns:
        config (dict): config dictionary with all parameters
    """

    # load config json file (should contain all necessary parameters)
    conf_parser = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    conf_parser.add_argument("-c", "--conf_file", default="configs/taylorgreen.json",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    with open(args.conf_file, 'r') as f:
        loaded_params = json.load(f)
    config = loaded_params
    
    # additional command-line arguments to modify config
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default=None, help='name tag')
    parser.add_argument('-e', '--example', type=str, default=None, 
                    help='example setup (module name under examples folder)')
    parser.add_argument('--N', type=int, default=None, help='grid size')
    parser.add_argument('--dt', type=float, default=None, help='time step size')
    parser.add_argument('--T', type=int, default=None, help='total time steps')
    parser.add_argument('--diff', type=float, default=None, help='diffusion coefficent')
    parser.add_argument('--visc', type=float, default=None, help='viscosity coefficent')
    parser.add_argument('--draw', type=str, default=None, choices=['velocity', 'density', 'curl', 'mix'])
    parser.add_argument('--fps', type=int, default=None, help="fps for saved animation")
    parser.add_argument('--save_grids', type=int, default=None, help="save grids")
    args = parser.parse_args(remaining_argv)

    for k, v in args.__dict__.items():
        if v is not None:
            config[k] = v
    
    # set command-line arguments as attributes
    print("----Configuration-----")
    for k, v in config.items():
        print("{0:20}".format(k), v)

    return config
