# Modemm Server

The Modemm server allows model requests to be run through it. The server can be run using the following:

`python3 -m modemm`

This will run the modemm server through port 14145 using the config.json file in the current directory.

### Server options

The server command provides the following options:

`-c` or `--config_file` can be used to specify a different config file than config.json

`-d` or `--dynamic_config` uses the dynamic config which allows the config file to be changed externally on-the-fly. This is useful for changing and debugging model parameters on demand.

`-p` or `--port` specifies the port the server runs on. By default, it runs on 14145

`-n` or `--no-advertise` tells the server not to expose paths or docs, which can reduce scraping and API abuse. It is recommended to set this if you are serving over the internet.

### Config options

To configure which models are served through the server, edit config.json with the models to be served. The syntax for models is as follows:

```json
{
  ...
  "models": ["name", "type", "path"],
  ...
}
```

`name` is what will be shown to the user being served.

`type` is the model type implemented by Modemm.

`path` is a relative or absolute path to the needed model files. For diffusers or transformers backend models, this can also be a huggingface repo.

The current types implemented are:

- absolutely nothing
