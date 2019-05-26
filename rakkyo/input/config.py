import pyhocon
import sys
import os, os.path, copy
import __main__


def parse_config_args() -> pyhocon.ConfigTree:
    main_file = __main__.__file__
    def_config = main_file.replace('.py', '.defaults')
    if os.path.exists(def_config):
        def_conf_obj = pyhocon.ConfigFactory.parse_file(def_config, resolve=False)
    else:
        def_conf_obj = pyhocon.ConfigTree()

    config_name = sys.argv[1]
    if not os.path.exists(config_name):
        raise pyhocon.ConfigException(f"config with file={config_name} does not exist")

    cur_conf_obj = pyhocon.ConfigFactory.parse_file(config_name, resolve=False)

    rest_config_str = '\n'.join(sys.argv[2:])
    rest_config = pyhocon.ConfigFactory.parse_string(rest_config_str, resolve=False)

    merged1 = pyhocon.ConfigTree.merge_configs(copy.deepcopy(cur_conf_obj), rest_config)
    merged2 = pyhocon.ConfigTree.merge_configs(copy.deepcopy(def_conf_obj), merged1)

    assembled_conf = copy.deepcopy(merged2)
    if pyhocon.ConfigParser.resolve_substitutions(assembled_conf):
        raise pyhocon.ConfigException("Failed to resolve config")
    return assembled_conf
