import yaml

try:
    with open('config.yml', 'r') as f:
        cfg = yaml.safe_load(f)
       # print ("loaded")
except FileNotFoundError:
    # Handle the case when the config file is not found
    cfg = None  # Or provide a default configuration
    

# Use cfg as needed
