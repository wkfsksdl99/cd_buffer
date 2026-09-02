from config import build_parser
from cd_buffer_core import run_cd_buffer_tta

def main_cd_buffer_tta(args):
    return run_cd_buffer_tta(args, light=False)
if __name__ == '__main__':
    main_cd_buffer_tta(build_parser(light=False).parse_args())

