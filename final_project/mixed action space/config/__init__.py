import imp

# if configuration parameter are missing: 
# raise attribute error and ask for set attribute value 
def set_missing_cf_field(field_name, cf, default_value):
    if not hasattr(cf, field_name):
        setattr(cf, field_name, default_value)

# load module config
def load_config(config_path):
    cf = imp.load_source('config', config_path)
    return cf
