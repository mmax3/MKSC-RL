import os

import gymnasium.envs.registration
from dotenv import load_dotenv

from GymBizHawk.gymbizhawk.bizhawk import BizHawkEnv

load_dotenv()

gymnasium.envs.registration.register(
    id="MKSC-v0",
    entry_point=__name__ + ":MKSC",
)


class MKSC(BizHawkEnv):
    def __init__(self, **kwargs):
        assert "BIZHAWK_DIR" in os.environ
        assert "MKSC_PATH" in os.environ  # used lua
        super().__init__(
            bizhawk_dir=os.environ["BIZHAWK_DIR"],
            lua_file=os.path.join(os.path.dirname(__file__), "mksc.lua"),
            mode="RUN",  # "RUN", "FAST_RUN", "RECORD", "DEBUG"
            observation_type="IMAGE",
            display_name="BizHawk-MKSC",
            setup_str_for_lua="1",  # option
            **kwargs,
        )
